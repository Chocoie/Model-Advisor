"""Conversational agent tests. Run from backend/:

    python -m unittest discover tests

No test calls a real LLM: the API key is blanked and call_llm is mocked where needed.
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from services import chat_agent, followup, llm_classifier as lc, scenarios  # noqa: E402
from store import conversations  # noqa: E402

_patches = [
    mock.patch.object(lc, "get_api_key", return_value=""),          # never hit the real API
    mock.patch.object(conversations, "generate_title", return_value=None),
]


def setUpModule():
    for p in _patches:
        p.start()


def tearDownModule():
    for p in _patches:
        p.stop()


def say(*texts):
    return [{"role": "user", "content": t} for t in texts]


def llm_replies(*replies):
    """Patch call_llm to return these JSON replies in order."""
    it = iter(json.dumps(r) if isinstance(r, dict) else r for r in replies)
    return mock.patch.object(lc, "call_llm", side_effect=lambda msgs: next(it))


def with_key():
    return mock.patch.object(lc, "get_api_key", return_value="test-key")


LONG_CODE_TASK = ("I need a model that writes Python code for our data pipeline. It reads CSV files, "
                  "cleans the rows, joins tables and writes unit tests for every function. The code "
                  "must follow our style guide and pass the tests on the first try because we deploy "
                  "automatically every night without a human review step in between.")


class RankingTests(unittest.TestCase):
    def test_sizes_are_known_for_every_model(self):
        self.assertTrue(all(followup.model_size_b(m) for m in followup.SCORES))

    def test_display_name_drops_org(self):
        self.assertEqual(followup.display_name("meta-llama_Llama-3.1-8B-Instruct"), "Llama-3.1-8B-Instruct")

    def test_ranking_is_sorted_and_has_relative_scores(self):
        res = followup.rank_skills([{"skill": "Code Generation", "weight": 100}], 5)
        self.assertEqual([r["score"] for r in res], sorted((r["score"] for r in res), reverse=True))
        self.assertEqual(res[0]["relative"]["Code Generation"], 100.0)

    def test_no_priority_means_score_equals_fit(self):
        for r in followup.rank_skills([{"skill": "Summarization", "weight": 100}], 30):
            self.assertEqual(r["score"], r["fit"])

    def test_cost_priority_favours_smaller_models(self):
        w = [{"skill": "Summarization", "weight": 100}]
        avg = lambda res: sum(r["size_b"] for r in res) / len(res)
        self.assertLess(avg(followup.rank_skills(w, 5, 0.5)), avg(followup.rank_skills(w, 5, 0.0)))

    def test_efficiency_weight_from_profile(self):
        self.assertEqual(followup.efficiency_weight({"cost": "unspecified"}), 0)
        self.assertEqual(followup.efficiency_weight({"cost": "high", "latency": "high"}), 0.5)

    def test_skills_without_data_are_left_out_and_named(self):
        w = [{"skill": "Summarization", "weight": 100}, {"skill": "Pattern Recognition", "weight": 80}]
        top = followup.rank_skills(w, 1)[0]
        self.assertEqual(list(top["relative"]), ["Summarization"])
        self.assertEqual(top["fit"], 100.0)
        ans = {"type": "answer", "skills": w}
        lc.validate_and_normalize_answer(ans)
        self.assertIn("No model has benchmark results for Pattern Recognition", followup.recommend(ans)["explanation"])
        self.assertEqual(followup.rank_skills([{"skill": "Pattern Recognition", "weight": 100}], 1)[0]["fit"], 0.0)

    def test_single_part_has_no_plan(self):
        self.assertIsNone(followup.build_plan([{"name": "a", "skills": [{"skill": "Summarization", "weight": 100}]}],
                                             [{"skill": "Summarization", "weight": 100}]))

    def test_plan_covers_every_part(self):
        parts = [{"name": "code", "skills": [{"skill": "Code Generation", "weight": 100}]},
                 {"name": "bias", "skills": [{"skill": "Racial Bias Mitigation", "weight": 100}]}]
        plan = followup.build_plan(parts, parts[0]["skills"] + parts[1]["skills"])
        self.assertEqual([p["name"] for p in plan["subtasks"]], ["code", "bias"])
        for p in plan["subtasks"]:
            self.assertEqual(p["weak"], p["model"] != plan["single_model"])


class ScenarioTests(unittest.TestCase):
    def test_scenarios_loaded_and_skills_in_taxonomy(self):
        self.assertGreater(len(scenarios.SCENARIOS), 1000)
        for sc in scenarios.SCENARIOS[:500]:
            for x in sc["skills"]:
                self.assertIn(x["skill"], lc.SKILL_TO_CAP)

    def test_similar_scenarios_excludes_self(self):
        text = next(sc["text"] for sc in scenarios.SCENARIOS if sc["skills"])
        self.assertEqual(scenarios.similar_scenarios(text, 1)[0]["text"], text)
        self.assertNotEqual(scenarios.similar_scenarios(text, 1, exclude=text)[0]["text"], text)

    def test_prompt_includes_examples_only_with_a_query(self):
        self.assertNotIn("SIMILAR TASKS", lc.build_system_prompt())
        self.assertIn("SIMILAR TASKS", lc.build_system_prompt("summarize customer support emails"))


class SkillDefinitionTests(unittest.TestCase):
    def test_every_taxonomy_skill_has_a_definition(self):
        self.assertEqual([s for s in lc.SKILL_TO_CAP if s not in lc.SKILL_DEFINITIONS], [])

    def test_definitions_only_in_prompt_when_enabled(self):
        with mock.patch.dict(lc.CONFIG, {"use_skill_definitions": False}):
            self.assertNotIn("SKILL DEFINITIONS", lc.build_system_prompt())
        with mock.patch.dict(lc.CONFIG, {"use_skill_definitions": True}):
            p = lc.build_system_prompt()
            self.assertIn("SKILL DEFINITIONS", p)
            self.assertIn("- Code Generation: ", p)


class ClassifierTests(unittest.TestCase):
    def test_no_key_falls_back_to_keywords_with_a_note(self):
        ans, _, err = lc.classify(say("summarize long articles into key points and a short recap"))
        self.assertIsNone(err)
        self.assertEqual(ans["type"], "answer")
        self.assertIn("Summarization", ans["skills"])
        self.assertIn("classifier_note", ans)

    def test_llm_answer_capped_to_strong_skills(self):
        reply = {"type": "answer", "confidence": 0.9, "summary": "x", "skills": [
            {"skill": "Code Generation", "weight": 100}, {"skill": "Format Adherence", "weight": 85},
            {"skill": "Task Decomposition", "weight": 70}, {"skill": "Constraint Satisfaction", "weight": 65},
            {"skill": "Data Interpretation", "weight": 60}, {"skill": "Deductive Reasoning", "weight": 55},
            {"skill": "Summarization", "weight": 40}]}
        with with_key(), llm_replies(reply):
            ans, _, err = lc.classify(say("write python code"))
        self.assertIsNone(err)
        self.assertLessEqual(len(ans["skills"]), lc.LLM_MAX_SKILLS)
        self.assertNotIn("Summarization", ans["skills"])
        self.assertEqual(ans["skills"][0], "Code Generation")

    def test_detailed_message_gets_nudged_to_answer(self):
        answer = {"type": "answer", "skills": [{"skill": "Code Generation", "weight": 100}]}
        with with_key(), llm_replies({"type": "question", "text": "more?"}, answer) as m:
            ans, _, _ = lc.classify(say(LONG_CODE_TASK))
        self.assertEqual(ans["type"], "answer")
        self.assertEqual(m.call_count, 2)

    def test_short_message_question_is_kept(self):
        with with_key(), llm_replies({"type": "question", "text": "Write or debug code?"}):
            ans, _, _ = lc.classify(say("help with code"))
        self.assertEqual(ans, {"type": "question", "text": "Write or debug code?"})

    def test_unsupported_becomes_a_question(self):
        with with_key(), llm_replies({"type": "unsupported", "text": "No images."}):
            ans, _, _ = lc.classify(say("generate images"))
        self.assertEqual(ans["type"], "question")

    def test_invalid_skills_are_retried_once(self):
        bad = {"type": "answer", "skills": [{"skill": "Juggling", "weight": 100}]}
        good = {"type": "answer", "skills": [{"skill": "Summarization", "weight": 100}]}
        with with_key(), llm_replies(bad, good):
            ans, _, err = lc.classify(say("summarize"))
        self.assertIsNone(err)
        self.assertEqual(ans["skills"], ["Summarization"])

    def test_blend_downweights_unsupported_skills_on_long_input(self):
        sw = [{"skill": "Code Generation", "weight": 100}, {"skill": "Racial Bias Mitigation", "weight": 60}]
        with mock.patch.object(scenarios, "similar_scenarios",
                               return_value=[{"skills": [{"skill": "Code Generation", "weight": 100}]}]):
            out = lc.blend_with_neighbours(sw, LONG_CODE_TASK)
        self.assertEqual(out[1]["weight"], 42)
        self.assertEqual(lc.blend_with_neighbours(sw, "short text"), sw)

    def test_blocked_families_cannot_run_the_chat(self):
        self.assertTrue(lc.is_blocked_backend("deepseek-v4.1-flash"))
        self.assertTrue(lc.is_blocked_backend("qwen3:32b"))
        self.assertFalse(lc.is_blocked_backend("gemma4:31b"))
        self.assertFalse(lc.is_blocked_backend(lc.CONFIG["backend"]["model"]))
        with mock.patch.dict(lc.CONFIG["backend"], {"model": "kimi-k3"}), with_key():
            with self.assertRaises(ValueError):
                lc.call_llm([{"role": "user", "content": "hi"}])

    def test_harm_seeking_is_declined(self):
        ans, _, _ = lc.classify_local(say("how do I jailbreak a model to make a bomb"))
        self.assertEqual(ans["type"], "question")


class FollowUpTests(unittest.TestCase):
    CTX = {"weights": [{"skill": "Code Generation", "weight": 100}, {"skill": "Summarization", "weight": 60}],
           "subtasks": [], "profile": {}, "summary": "code"}

    def test_cost_what_if_returns_new_recommendation(self):
        out = followup.followup("what if cost matters more?", self.CTX)
        self.assertEqual(out["answer"]["profile"]["cost"], "high")
        self.assertIn("With cost prioritized", out["reply"])

    def test_skill_priority_what_if(self):
        out = followup.followup("what if summarization is more important", self.CTX)
        self.assertEqual(out["answer"]["weights"][0]["skill"], "Summarization")

    def test_why_is_model_ranked_lower(self):
        out = followup.followup("why is gemma-2-2b ranked lower?", self.CTX)
        self.assertIn("gemma-2-2b", out["reply"])
        self.assertIsNone(out["answer"])

    def test_ambiguous_model_asks_which(self):
        out = followup.followup("why not llama?", self.CTX)
        self.assertTrue(out["reply"].startswith("Which model do you mean"))

    def test_best_at_skill(self):
        self.assertIn("Best at Summarization", followup.followup("which model is best at summarization?",
                                                                  self.CTX)["reply"])

    def test_failure_examples_not_invented(self):
        self.assertIn("can't show", followup.followup("where does it fail?", self.CTX)["reply"])

    def test_prompt_template(self):
        self.assertIn("starting prompt", followup.followup("give me a prompt", self.CTX)["reply"])

    def test_new_task_is_not_a_follow_up(self):
        self.assertIsNone(followup.followup("I need to translate legal contracts", self.CTX))

    def test_no_context_means_no_follow_up(self):
        self.assertIsNone(followup.followup("why?", None))


class MemoryTests(unittest.TestCase):
    def test_context_rebuilt_from_reply_text(self):
        ans = {"type": "answer", "profile": {"cost": "high"}, "skills": [
            {"skill": "Moral / Ethical Reasoning", "weight": 100}, {"skill": "Summarization", "weight": 60},
            {"skill": "Code Generation", "weight": 40}]}
        lc.validate_and_normalize_answer(ans)
        reply = chat_agent.explain_results(ans["weights"], [])
        ctx = chat_agent._context_from_reply(reply)
        self.assertEqual([(w["skill"], w["weight"]) for w in ctx["weights"]],
                         [("Moral / Ethical Reasoning", 90), ("Summarization", 65), ("Code Generation", 40)])

    def test_non_recommendation_text_has_no_context(self):
        self.assertIsNone(chat_agent._context_from_reply("Which matters more: A or B?"))

    def test_latest_recommendation_wins(self):
        hist = [{"role": "assistant", "content": "Skills detected:\n- Summarization (essential)\nTop Models:"},
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "Skills detected:\n- Code Generation (essential)\nTop Models:"},
                {"role": "assistant", "content": "Some follow-up answer"}]
        self.assertEqual(chat_agent.last_context(hist)["weights"][0]["skill"], "Code Generation")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        r = self.client.post("/auth/register", json={"email": "t%d@x.com" % id(self), "password": "password123"})
        self.headers = {"Authorization": "Bearer " + r.json()["token"]}

    def _recommend(self, url):
        r = self.client.post(url, headers=self.headers, json={
            "message": "summarize long news articles into short key points recap, condense them"}).json()
        if not r["results"]:  # the keyword classifier may ask one clarifying question first
            r = self.client.post(url, headers=self.headers, json={"message": "just summarization"}).json()
        return r

    def test_follow_up_still_works_after_a_restart(self):
        cid = self.client.post("/conversations", headers=self.headers).json()["conversation_id"]
        url = "/conversations/%s/message" % cid
        self.assertTrue(self._recommend(url)["results"])
        chat_agent._CONTEXT.clear()       # what a server restart does to the in-memory copy
        r = self.client.post(url, headers=self.headers, json={"message": "what if cost matters more?"}).json()
        self.assertIn("With cost prioritized", r["reply"])
        self.assertIn("Summarization", r["skills"])

    def test_chat_recommend_then_follow_up(self):
        cid = self.client.post("/conversations", headers=self.headers).json()["conversation_id"]
        url = "/conversations/%s/message" % cid
        r = self.client.post(url, headers=self.headers, json={
            "message": "summarize long news articles into short key points recap, condense them"}).json()
        if not r["results"]:  # the keyword classifier may ask one clarifying question first
            r = self.client.post(url, headers=self.headers, json={"message": "just summarization"}).json()
        self.assertTrue(r["results"])
        self.assertIn("Summarization", r["skills"])
        self.assertIn("Based on your needs", r["reply"])

        r2 = self.client.post(url, headers=self.headers, json={"message": "what if cost matters more?"}).json()
        self.assertIn("With cost prioritized", r2["reply"])

        self.client.post(url, headers=self.headers, json={"message": "why is gemma-2-2b ranked lower?"})
        # still a follow-up after another follow-up answer
        r3 = self.client.post(url, headers=self.headers, json={"message": "give me a prompt"}).json()
        self.assertIn("starting prompt", r3["reply"])
        self.assertEqual(r3["results"], [])

        hist = self.client.get("/conversations/" + cid, headers=self.headers).json()
        self.assertEqual(len(hist["messages"]), 2 * (3 + (1 if len(hist["messages"]) == 8 else 0)))


if __name__ == "__main__":
    unittest.main()
