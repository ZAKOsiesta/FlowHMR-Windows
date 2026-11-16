"""Exercise the native launcher's WebView2 prerequisite decision on Windows."""
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless


ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('WINDIR', '')) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'


@skipUnless(os.name == 'nt' and COMPILER.is_file(), 'Requires the Windows launcher compiler')
class DesktopPrerequisitesTest(TestCase):
    def test_transient_webview_registration_requires_installation(self):
        harness = '''
using System;
using System.Reflection;
internal static class CheckPrerequisites {
    public static int Main() {
        var check = typeof(FlowHMRLauncher).GetMethod("InstalledWebView2Version",
            BindingFlags.Static | BindingFlags.NonPublic);
        foreach (string value in new string[] { null, "", "0.0.0.0", "157.0.4341.42.deleting", "not-a-version" })
            if ((bool)check.Invoke(null, new object[] { value })) return 1;
        foreach (string value in new string[] { "157.0.4341.42", "109.0.1518.78" })
            if (!(bool)check.Invoke(null, new object[] { value })) return 2;
        return 0;
    }
}
'''
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'CheckPrerequisites.cs'
            source.write_text(harness, encoding='utf8')
            executable = directory / 'CheckPrerequisites.exe'
            subprocess.run([str(COMPILER), '/nologo', '/target:exe', '/platform:x64',
                            '/main:CheckPrerequisites', '/reference:System.Windows.Forms.dll',
                            f'/out:{executable}', str(source),
                            str(ROOT / 'tools/windows/FlowHMRLauncher.cs'),
                            str(ROOT / 'tools/windows/DesktopUpdate.cs')], check=True, capture_output=True)
            result = subprocess.run([str(executable)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
