from pathlib import Path
import copy
import hashlib
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from c4_rule_followup import aggregate, build, expected, group_keys, prompt, record_id


class FollowupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = json.loads((ROOT/"manifests/c4_rule_followup_v2_seed42_20260905.json").read_text())

    def test_counts_and_partition(self):
        m=self.m
        self.assertEqual(m["n_oracle_queries"],81)
        self.assertEqual(m["n_semantic_valid_oracle_queries"],69)
        self.assertEqual(m["n_menu_groups"],23)
        self.assertEqual(len(expected(m)),1071)
        parts=[{x for x in expected(m) if x[1]%3==i} for i in range(3)]
        self.assertEqual(sum(map(len,parts)),len(set.union(*parts)))

    def test_prompts_and_candidate_identity(self):
        for arm,k in (("oracle","oracle_groups"),("menu","menu_groups")):
            for g in self.m[k]:
                for key in group_keys(arm,g):
                    s=prompt(self.m,g,*key[2:])
                    self.assertEqual(hashlib.sha256(s.encode()).hexdigest(),g["prompt_hashes"][record_id(key)])
                    self.assertEqual(s.count("<STUDENT_STATE>"),1)
                    query=s.split("<query>")[1]
                    self.assertNotIn("Correct answer:",query)
                    if arm=="oracle":
                        self.assertNotIn("Valid outputs:",query)
                        self.assertNotIn("Trial 1:",s)
                if arm=="menu":
                    self.assertTrue(all(c["example_id"].endswith(":malrule_next_response") for c in g["contexts"].values()))
                    self.assertEqual(len({c["seg_history"] for c in g["contexts"].values()}),len(g["contexts"]))
                    self.assertEqual(len({c["seg_instruction"] for c in g["contexts"].values()}),1)
                    self.assertNotEqual(g["queries"][0]["donor_target"],g["queries"][1]["donor_target"])
                    for o in (0,1):
                        menus=[prompt(self.m,g,q,o,"swap").split("Valid outputs:")[1] for q in (0,1,-1)]
                        self.assertEqual(menus[0],menus[1]); self.assertEqual(menus[1],menus[2])
                    self.assertEqual(g["menus"][0],list(reversed(g["menus"][1])))
                    for q in g["queries"]:
                        self.assertTrue(q["semantic_valid"])
                        self.assertNotEqual(q["random_rule_target"],q["donor_target"])

    def test_endpoint_and_corruption_rejection(self):
        rs=[]
        for arm,k in (("oracle","oracle_groups"),("menu","menu_groups")):
            for g in self.m[k]:
                for key in group_keys(arm,g):
                    qi,_,c=key[2:]
                    q=g["queries"][max(qi,0)]
                    y=q["donor_target"] if c in ("donor_rule","swap","same_donor_rule") else q["recipient_target"]
                    rs.append({"key":key,"prompt_sha256":g["prompt_hashes"][record_id(key)],"text":f"<predicted_response>{y}</predicted_response>","parsed":y,"parse_type":"tagged","stop":"closing_tag"})
        s=aggregate(self.m,rs,bootstrap=20)
        self.assertEqual(s["menu"]["all"]["contrasts"]["joint_swap_minus_random_real"]["estimate"],1)
        self.assertEqual(s["oracle"]["all_legacy81"]["contrasts"]["donor_minus_no_rule"]["estimate"],1)
        with self.assertRaises(AssertionError): aggregate(self.m,rs[:-1],bootstrap=20)
        with self.assertRaises(AssertionError): aggregate(self.m,rs+rs[:1],bootstrap=20)
        bad=copy.deepcopy(rs); bad[0]["text"]="42"
        with self.assertRaises(AssertionError): aggregate(self.m,bad,bootstrap=20)


if __name__=="__main__":
    unittest.main()
