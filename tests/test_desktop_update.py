"""Run the native file transaction with real Windows rename locks."""
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('WINDIR', '')) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'

HARNESS = r'''
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.Win32.SafeHandles;
internal static class UpdateScenario {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern SafeFileHandle CreateFile(string path, uint access, uint share,
        IntPtr security, uint creation, uint flags, IntPtr template);
    static void Write(string root, string path, string value) {
        string file = Path.Combine(root, path);
        Directory.CreateDirectory(Path.GetDirectoryName(file));
        File.WriteAllText(file, value);
    }
    static void Expect(bool ok, string message) { if (!ok) throw new Exception(message); }
    static SafeFileHandle Lock(string path) {
        var handle = CreateFile(path, 0x80000000, 1, IntPtr.Zero, 3, 0, IntPtr.Zero);
        if (handle.IsInvalid) throw new System.ComponentModel.Win32Exception();
        return handle;
    }
    static void WithTransientLock(string path, Action action) {
        using (var handle = Lock(path)) {
            var release = new Thread(() => { Thread.Sleep(1400); handle.Dispose(); });
            release.Start();
            try { action(); } finally { release.Join(); }
        }
    }
    public static int Main(string[] args) {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        try {
            string root = args[0], mode = args[1];
            string work = Path.Combine(root, ".cache/program-update");
            Write(root, "FlowHMR-Version.txt", "0.3.6");
            Write(root, "third_party/original.txt", "old dependency");
            Write(root, "tools/original.txt", "old code");
            Write(root, "models/keep.txt", "user model");
            Write(work, "ready/files/FlowHMR-Version.txt", "9.9.903");
            Write(work, "ready/files/third_party/new.txt", "new dependency");
            Write(work, "ready/files/tools/new.txt", "new code");
            Write(work, "ready/targets.txt", "tools\nthird_party\nFlowHMR-Version.txt\n");
            var update = new DesktopUpdate(root);
            if (mode == "apply-lock")
                WithTransientLock(Path.Combine(work, "ready/files/third_party/new.txt"),
                    () => Expect(update.Prepare(), "Temporary lock cancelled the update"));
            else Expect(update.Prepare(), "Update did not start");
            Write(work, "applying/healthy", "");
            Expect(update.Finish(), "Healthy update not committed");
            Expect(File.ReadAllText(Path.Combine(root, "FlowHMR-Version.txt")) == "9.9.903", "Version changed");
            Write(work, "rollback-request.txt", "0.3.6");
            Action rollback = () => Expect(!update.Prepare(), "Rollback unexpectedly staged an update");
            if (mode == "rollback-lock" || mode == "resume")
                Directory.Move(Path.Combine(work, "previous"), Path.Combine(work, "undoing"));
            if (mode == "rollback-lock")
                WithTransientLock(Path.Combine(work, "undoing/backup/third_party/original.txt"), rollback);
            else if (mode == "resume") {
                using (var handle = Lock(Path.Combine(work, "undoing/backup/third_party/original.txt"))) {
                    bool failed = false;
                    try { rollback(); }
                    catch (IOException) { failed = true; }
                    catch (UnauthorizedAccessException) { failed = true; }
                    Expect(failed, "A persistent lock was ignored");
                    Expect(Directory.Exists(Path.Combine(work, "undoing")), "Recovery state was discarded");
                }
                rollback();
            } else rollback();
            Expect(File.ReadAllText(Path.Combine(root, "FlowHMR-Version.txt")) == "0.3.6", "Old version not restored");
            Expect(File.ReadAllText(Path.Combine(root, "third_party/original.txt")) == "old dependency", "Old dependency lost");
            Expect(File.ReadAllText(Path.Combine(root, "tools/original.txt")) == "old code", "Old code lost");
            Expect(!File.Exists(Path.Combine(root, "third_party/new.txt")), "New file leaked into old version");
            Expect(File.ReadAllText(Path.Combine(root, "models/keep.txt")) == "user model", "User model changed");
            Expect(Directory.Exists(Path.Combine(work, "forward")), "Forward recovery missing");
            Console.WriteLine("PASS " + mode);
            return 0;
        } catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
'''


@skipUnless(os.name == 'nt' and COMPILER.is_file(), 'Requires the Windows launcher compiler')
class DesktopTransactionTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        source = cls.directory / 'UpdateScenario.cs'
        source.write_text(HARNESS, encoding='utf8')
        cls.executable = cls.directory / 'UpdateScenario.exe'
        subprocess.run([str(COMPILER), '/nologo', '/target:exe', f'/out:{cls.executable}',
                        str(source), str(ROOT / 'tools/windows/DesktopUpdate.cs')],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_scenario(self, mode):
        result = subprocess.run([str(self.executable), str(self.directory / mode), mode],
                                capture_output=True, text=True, encoding='utf8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_upgrade_and_manual_rollback(self):
        self.run_scenario('roundtrip')

    def test_temporary_lock_during_upgrade(self):
        self.run_scenario('apply-lock')

    def test_temporary_lock_during_rollback(self):
        self.run_scenario('rollback-lock')

    def test_interrupted_rollback_resumes_without_losing_files(self):
        self.run_scenario('resume')
