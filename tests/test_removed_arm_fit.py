"""Historical jobs must not reactivate the removed arm optimizer on rerun."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from web import server
from flowhmr.mcp.contract import Parameters


class RemovedArmFitTests(unittest.TestCase):
    def test_rerun_of_old_enabled_job_cannot_enable_arm_fitting(self):
        for source,base in [({'visible_arm_fit':True},None),({}, {'visible_arm_fit':True})]:
            self.assertFalse(server.parse_params(source,base).get('visible_arm_fit',False))

    def test_new_defaults_and_public_schema_do_not_offer_arm_fitting(self):
        self.assertFalse(server.default_params().get('visible_arm_fit',False))
        self.assertNotIn('visible_arm_fit',Parameters.model_fields)


if __name__=='__main__':unittest.main()
