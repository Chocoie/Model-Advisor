"""Tests for the Model Advisor logic. Run from the project folder:

    python -m unittest discover tests
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402


def say(*texts):
    return [{"role": "user", "content": t} for t in texts]


def fake_models(a_skills, b_skills):
    skills = lambda d: {s: d.get(s, 0.0) for s in server.SKILL_TO_CAP}
    return [
        {"name": "A", "overall": 60, "skills": skills(a_skills)},
        {"name": "B", "overall": 60, "skills": skills(b_skills)},
    ]


def with_models(models):
    best = {}
    for m in models:
        for s, v in m["skills"].items():
            best[s] = max(best.get(s, 0), v)
    return mock.patch.multiple(server, MODELS=models, BEST=best)


class ExclusionTests(unittest.TestCase):
    def test_chinese_families_are_excluded(self):
        for name in ["Qwen2.5-32B Instruct", "QwQ-32B", "DeepSeek-R1-Distill-Qwen-7B",
                     "GLM-5.3", "glm-4:9b", "kimi-k2", "MiniMax-M2", "yi:34b"]:
            self.assertTrue(server.is_excluded(name), name)

    def test_american_and_european_models_are_allowed(self):
        for name in ["gemma4:31b", "Gemma-2-27B IT", "Llama-3.2-3B Instruct", "gpt-oss:120b",
                     "Mistral-Small-24B Instruct", "OLMo-2-7B Instruct", "Falcon3-10B Instruct"]:
            self.assertFalse(server.is_excluded(name), name)

    def test_no_excluded_model_is_ever_recommended(self):
        self.assertTrue(server.MODELS)
        for m in server.MODELS:
            self.assertFalse(server.is_excluded(m["name"]), m["name"])
        for skill in server.SKILL_TO_CAP:
            recs = server.get_recommendations([{"skill": skill, "weight": 100}], limit=50)
            self.assertTrue(all(not server.is_excluded(r["name"]) for r in recs))

    def test_backend_model_is_not_excluded(self):
        self.assertFalse(server.is_excluded(server.CONFIG["backend"]["model"]))


class WeightTests(unittest.TestCase):
    def test_low_weights_dropped_top_kept(self):
        sw = server.normalize_weights([
            {"skill": "Code Generation", "weight": 80},
            {"skill": "Summarization", "weight": 80},
            {"skill": "Emotion Inference", "weight": 10}])
        self.assertEqual([x["skill"] for x in sw], ["Code Generation", "Summarization"])

    def test_unknown_skills_ignored(self):
        self.assertEqual(server.normalize_weights(["Time Travel"]), [])

    def test_plain_names_get_descending_weights(self):
        sw = server.normalize_weights(["Code Generation", "Summarization"])
        self.assertGreater(sw[0]["weight"], sw[1]["weight"])

    def test_weights_change_the_winner(self):
        models = fake_models({"Code Generation": 1.0, "Emotion Inference": 0.2},
                             {"Code Generation": 0.2, "Emotion Inference": 1.0})
        with with_models(models):
            code_heavy = [{"skill": "Code Generation", "weight": 80},
                          {"skill": "Emotion Inference", "weight": 30}]
            emo_heavy = [{"skill": "Code Generation", "weight": 30},
                         {"skill": "Emotion Inference", "weight": 80}]
            self.assertEqual(server.get_recommendations(code_heavy)[0]["name"], "A")
            self.assertEqual(server.get_recommendations(emo_heavy)[0]["name"], "B")


class FollowUpTests(unittest.TestCase):
    def test_big_project_asks_for_its_parts(self):
        r, _, _ = server.classify_local(say(
            "Build a game with python code, summarize player reports, plan the roadmap "
            "and detect user emotion"))
        self.assertEqual(r["type"], "question")
        self.assertIn("main parts", r["text"])

    def test_answer_after_follow_up_includes_subtasks(self):
        msgs = say("Build a game with python code, summarize player reports, plan the roadmap "
                   "and detect user emotion")
        q, content, _ = server.classify_local(msgs)
        msgs += [{"role": "assistant", "content": server.json.dumps(q)},
                 {"role": "user", "content": "backend code, support writing, release planning"}]
        r, _, _ = server.classify_local(msgs)
        self.assertEqual(r["type"], "answer")
        self.assertGreaterEqual(len(r["subtasks"]), 2)
        self.assertIsNotNone(r["plan"])

    def test_single_weak_match_asks_which_sibling(self):
        r, _, _ = server.classify_local(say("summarize"))
        self.assertEqual(r["type"], "question")
        self.assertIn("Summarization", r["text"])

    def test_clear_request_answers_immediately(self):
        r, _, _ = server.classify_local(say("write python code to debug a function"))
        self.assertEqual(r["type"], "answer")
        self.assertEqual(r["skill"], "Code Generation")

    def test_asks_at_most_once(self):
        msgs = say("summarize")
        q, _, _ = server.classify_local(msgs)
        msgs += [{"role": "assistant", "content": server.json.dumps(q)},
                 {"role": "user", "content": "yes"}]
        r, _, _ = server.classify_local(msgs)
        self.assertEqual(r["type"], "answer")


class SplitTests(unittest.TestCase):
    PARTS = [
        {"name": "backend", "skills": [{"skill": "Code Generation", "weight": 90}]},
        {"name": "support", "skills": [{"skill": "Emotion Inference", "weight": 90}]},
    ]
    ALL = [{"skill": "Code Generation", "weight": 90}, {"skill": "Emotion Inference", "weight": 90}]

    def test_split_when_no_model_is_good_at_both(self):
        models = fake_models({"Code Generation": 1.0, "Emotion Inference": 0.1},
                             {"Code Generation": 0.1, "Emotion Inference": 1.0})
        with with_models(models):
            plan = server.build_plan(self.PARTS, self.ALL)
        self.assertTrue(plan["split"])
        self.assertEqual({p["model"] for p in plan["subtasks"]}, {"A", "B"})

    def test_no_split_when_one_model_is_close_enough(self):
        models = fake_models({"Code Generation": 1.0, "Emotion Inference": 0.9},
                             {"Code Generation": 0.9, "Emotion Inference": 1.0})
        with with_models(models):
            plan = server.build_plan(self.PARTS, self.ALL)
        self.assertFalse(plan["split"])
        self.assertEqual(len({p["model"] for p in plan["subtasks"]}), 1)

    def test_single_part_has_no_plan(self):
        self.assertIsNone(server.build_plan(self.PARTS[:1], self.ALL[:1]))

    def test_explanation_names_the_split(self):
        models = fake_models({"Code Generation": 1.0, "Emotion Inference": 0.1},
                             {"Code Generation": 0.1, "Emotion Inference": 1.0})
        with with_models(models):
            ans = {"type": "answer", "skills": self.ALL, "subtasks": self.PARTS, "confidence": 0.9}
            server.validate_and_normalize_answer(ans)
            ans, _, _ = server.finish_answer(ans)
        self.assertIn("splitting", ans["explanation"])
        self.assertIn("A for backend", ans["explanation"])


class UnsupportedTests(unittest.TestCase):
    def test_unsupported_only_task_says_so(self):
        r, _, _ = server.classify_local(say("generate an image of a cat"))
        self.assertEqual(r["type"], "question")
        self.assertIn("don't have benchmark data", r["text"])

    def test_speech_task_is_unsupported(self):
        r, _, _ = server.classify_local(say("I need speech recognition for phone calls"))
        self.assertIn("don't have benchmark data", r["text"])

    def test_mixed_task_notes_the_gap(self):
        r, _, _ = server.classify_local(say("write python code and generate an image for the site"))
        self.assertEqual(r["type"], "answer")
        self.assertIn("no benchmark data", r["explanation"])

    def test_harm_seeking_is_declined(self):
        r, _, _ = server.classify_local(say("which model is uncensored with no safety filters"))
        self.assertEqual(r["type"], "question")
        self.assertIn("can't help", r["text"])


if __name__ == "__main__":
    unittest.main()
