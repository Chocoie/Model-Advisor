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


def fake_models(a_skills, b_skills, names=("A", "B")):
    skills = lambda d: {s: d.get(s, 0.0) for s in server.SKILL_TO_CAP}
    return [
        {"name": names[0], "overall": 60, "skills": skills(a_skills)},
        {"name": names[1], "overall": 60, "skills": skills(b_skills)},
    ]


def with_models(models):
    best = {}
    for m in models:
        for s, v in m["skills"].items():
            best[s] = max(best.get(s, 0), v)
    return mock.patch.multiple(server, MODELS=models, BEST=best)


class BackendBlockTests(unittest.TestCase):
    """Chinese-developed families may be recommended but never run the chatbot itself."""

    def test_blocked_families_cannot_be_the_backend(self):
        for name in ["Qwen2.5-32B Instruct", "QwQ-32B", "DeepSeek-R1-Distill-Qwen-7B",
                     "GLM-5.3", "glm-4:9b", "kimi-k2", "MiniMax-M2", "yi:34b"]:
            self.assertTrue(server.is_excluded(name), name)

    def test_american_and_european_models_may_be_the_backend(self):
        for name in ["gemma4:31b", "Gemma-2-27B IT", "Llama-3.2-3B Instruct", "gpt-oss:120b",
                     "Mistral-Small-24B Instruct", "OLMo-2-7B Instruct", "Falcon3-10B Instruct"]:
            self.assertFalse(server.is_excluded(name), name)

    def test_blocked_models_can_still_be_recommended(self):
        names = [m["name"] for m in server.MODELS]
        self.assertTrue(any(server.is_excluded(n) for n in names))
        recs = server.get_recommendations([{"skill": "Causal Reasoning", "weight": 100}], limit=50)
        self.assertEqual(len(recs), len(server.MODELS))

    def test_configured_backend_is_not_blocked(self):
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



def ctx_from(skills, profile=None):
    ans = {"type": "answer", "skills": skills, "subtasks": [], "profile": profile or {},
           "confidence": 0.9, "summary": "test"}
    server.validate_and_normalize_answer(ans)
    return server.finish_answer(ans)[0]


class ProfileTests(unittest.TestCase):
    def test_profile_levels_default_to_unspecified(self):
        p = server.normalize_profile({"cost": "HIGH", "safety": "bogus", "domain": " law "})
        self.assertEqual((p["cost"], p["safety"], p["domain"]), ("high", "unspecified", "law"))

    def test_local_classifier_picks_up_stated_priorities(self):
        r, _, _ = server.classify_local(say("a cheap and fast python code helper"))
        self.assertEqual(r["profile"]["cost"], "high")
        self.assertEqual(r["profile"]["latency"], "high")
        self.assertEqual(r["profile"]["safety"], "unspecified")

    def test_profile_survives_recompute_from_edits(self):
        ans = ctx_from([{"skill": "Code Generation", "weight": 90}], {"domain": "games"})
        r, _, err = server.recommend_from_payload(
            {"skills": ans["weights"], "profile": {"domain": "games", "cost": "high"}})
        self.assertIsNone(err)
        self.assertEqual(r["profile"]["domain"], "games")
        self.assertGreater(r["efficiency_weight"], 0)

    def test_bad_edit_is_rejected(self):
        r, _, err = server.recommend_from_payload({"skills": ["Nonsense"]})
        self.assertIsNone(r)
        self.assertTrue(err)


class EfficiencyTests(unittest.TestCase):
    def models(self):
        a, b = fake_models({"Code Generation": 1.0}, {"Code Generation": 0.8})
        a["size_b"], b["size_b"] = 32, 0.5
        return [a, b]

    def test_cost_priority_shifts_ranking_to_smaller_model(self):
        sw = [{"skill": "Code Generation", "weight": 100}]
        with with_models(self.models()):
            self.assertEqual(server.get_recommendations(sw, 1, 0.0)[0]["name"], "A")
            lam = server.efficiency_weight({"cost": "high", "latency": "unspecified"})
            self.assertEqual(server.get_recommendations(sw, 1, lam)[0]["name"], "B")

    def test_no_priority_means_no_shift(self):
        self.assertEqual(server.efficiency_weight(server.normalize_profile({})), 0)


class FollowUpAnswerTests(unittest.TestCase):
    SKILLS = [{"skill": "Code Generation", "weight": 90}, {"skill": "Emotion Inference", "weight": 60}]

    def models(self):
        a, b = fake_models({"Code Generation": 1.0, "Emotion Inference": 0.8},
                           {"Code Generation": 0.8, "Emotion Inference": 1.0},
                           names=("Alpha", "Beta"))
        a["size_b"], b["size_b"] = 32, 0.5
        return [a, b]

    def test_cost_what_if_changes_the_recommendation(self):
        with with_models(self.models()):
            ctx = ctx_from(self.SKILLS)
            r, _, _ = server.followup("What if cost is more important to me?", ctx)
        self.assertEqual(r["type"], "answer")
        self.assertEqual(r["profile"]["cost"], "high")
        self.assertIn("changes from Alpha to Beta", r["explanation"])

    def test_skill_priority_what_if(self):
        with with_models(self.models()):
            ctx = ctx_from(self.SKILLS)
            r, _, _ = server.followup("what if emotion inference is more important", ctx)
        self.assertEqual(r["weights"][0]["skill"], "Emotion Inference")
        self.assertEqual(r["weights"][0]["weight"], 100)

    def test_why_is_model_ranked_lower(self):
        with with_models(self.models()):
            ctx = ctx_from(self.SKILLS)
            r, _, _ = server.followup("why is Beta ranked second?", ctx)
        self.assertEqual(r["type"], "message")
        self.assertIn("#2", r["text"])

    def test_which_model_best_at_skill(self):
        with with_models(self.models()):
            ctx = ctx_from(self.SKILLS)
            r, _, _ = server.followup("which model was best at emotion inference?", ctx)
        self.assertTrue(r["text"].startswith("Best at Emotion Inference: Beta"))

    def test_failure_examples_are_not_invented(self):
        with with_models(self.models()):
            ctx = ctx_from(self.SKILLS)
            r, _, _ = server.followup("show me where Alpha failed", ctx)
        self.assertIn("can't show specific failed examples", r["text"])

    def test_prompt_template(self):
        ctx = ctx_from(self.SKILLS, {"domain": "games"})
        r, _, _ = server.followup("what prompt should I use?", ctx)
        self.assertIn("starting prompt", r["text"])
        self.assertIn("games", r["text"])

    def test_new_task_is_not_treated_as_follow_up(self):
        ctx = ctx_from(self.SKILLS)
        self.assertIsNone(server.followup("translate legal contracts into plain english", ctx))

    def test_respond_falls_through_to_classification(self):
        ctx = ctx_from(self.SKILLS)
        r, _, _ = server.respond(say("write python code to debug a function"), ctx)
        self.assertEqual(r["type"], "answer")


class AmbiguityAndGateTests(unittest.TestCase):
    def test_ambiguous_model_name_asks_which_one(self):
        ctx = ctx_from([{"skill": "Code Generation", "weight": 90}])
        r, _, _ = server.followup("why is Gemma ranked lower?", ctx)
        self.assertEqual(r["type"], "message")
        self.assertTrue(r["text"].startswith("Which model do you mean"))

    def test_weak_small_model_never_wins_on_cost_alone(self):
        sw = [{"skill": "Code Generation", "weight": 100}]
        top = server.get_recommendations(sw, 3, 0.5)
        self.assertTrue(all(r["fit"] >= 50 for r in top[:1]))
        self.assertNotIn("Qwen2.5-0.5B Instruct", [r["name"] for r in top[:1]])


class ClassifierModeTests(unittest.TestCase):
    def test_llm_is_the_default_classifier(self):
        self.assertFalse(server.use_local_classifier())

    def test_no_key_falls_back_to_keywords_with_a_note(self):
        with mock.patch.object(server, "get_api_key", return_value=""):
            r, _, _ = server.classify(say("write python code to debug a function"))
        self.assertEqual(r["type"], "answer")
        self.assertIn("no API key", r["classifier_note"])

    def test_llm_answer_is_validated_and_gets_recommendations(self):
        reply = server.json.dumps({
            "type": "answer", "confidence": 0.8, "summary": "grading",
            "skills": [{"skill": "Factual Content Generation", "weight": 90},
                       {"skill": "Made Up Skill", "weight": 80}],
            "profile": {"domain": "education", "accuracy": "high", "cost": "medium"}})
        with mock.patch.object(server, "get_api_key", return_value="k"), \
                mock.patch.object(server, "call_llm", return_value=reply):
            r, _, err = server.classify(say("grade physics essays"))
        self.assertIsNone(err)
        self.assertEqual(r["skills"], ["Factual Content Generation"])
        self.assertEqual(r["profile"]["domain"], "education")
        self.assertEqual(len(r["recommendations"]), 3)

    def test_llm_question_and_unsupported_both_become_messages_to_user(self):
        for kind in ("question", "unsupported"):
            reply = server.json.dumps({"type": kind, "text": "what language?"})
            with mock.patch.object(server, "get_api_key", return_value="k"), \
                    mock.patch.object(server, "call_llm", return_value=reply):
                r, _, _ = server.classify(say("translate"))
            self.assertEqual(r, {"type": "question", "text": "what language?"})


if __name__ == "__main__":
    unittest.main()
