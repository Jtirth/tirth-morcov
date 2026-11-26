import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app import app


class FlaskApiTests(unittest.TestCase):
    def setUp(self):
        app.config.pop("TRAINING_TEXT", None)
        self.client = app.test_client()

    def tearDown(self):
        app.config.pop("TRAINING_TEXT", None)

    def test_first_and_second_order_probabilities(self):
        app.config["TRAINING_TEXT"] = "one red apple. one blue sky. one red fruit."
        first = self.client.get("/api/predict?text=one&order=1").get_json()
        second = self.client.get("/api/predict?text=one%20red&order=2").get_json()
        self.assertEqual(first["context"], ["one"])
        self.assertEqual(first["observations"], 3)
        self.assertEqual(first["predictions"][0]["word"], "red")
        self.assertAlmostEqual(first["predictions"][0]["probability"], 2 / 3)
        self.assertAlmostEqual(sum(row["probability"] for row in first["distribution"]), 1)
        self.assertEqual(second["context"], ["one", "red"])
        self.assertEqual(second["observations"], 2)
        self.assertAlmostEqual(sum(row["probability"] for row in second["distribution"]), 1)

    def test_train_dashboard_and_reset(self):
        trained = self.client.post("/api/train", json={"text": "blue bird. blue sky."})
        self.assertEqual(trained.status_code, 200)
        self.assertEqual(trained.get_json()["stats"]["total_words"], 4)
        stats = self.client.get("/api/dashboard").get_json()
        self.assertEqual(stats["top_words"][0]["word"], "blue")
        reset = self.client.post("/api/reset")
        self.assertEqual(reset.status_code, 200)
        self.assertNotEqual(self.client.get("/api/dashboard").get_json()["total_words"], 4)

    def test_invalid_and_empty_inputs(self):
        self.assertEqual(self.client.post("/api/train", json={"text": "   "}).status_code, 400)
        self.assertEqual(self.client.post("/api/train", json={"text": 12}).status_code, 400)
        self.assertEqual(self.client.get("/api/predict?text=hello&order=3").status_code, 400)
        self.assertEqual(self.client.get("/api/predict?text=x&order=1&limit=999").status_code, 200)
        self.assertEqual(self.client.get("/api/predict?text=" + "a" * 2001).status_code, 400)

    def test_comparison_uses_held_out_file_and_classroom_example(self):
        comparison = self.client.get("/api/compare").get_json()
        self.assertIn("1", comparison)
        self.assertIn("2", comparison)
        example = self.client.get("/api/example").get_json()
        self.assertEqual(example["observations"], 3)
        probabilities = {row["word"]: row["probability"] for row in example["distribution"]}
        self.assertEqual(probabilities, {"python": 1 / 3, "programming": 1 / 3, "coding": 1 / 3})

    def test_pages_health_and_security_headers(self):
        for path in ("/", "/dashboard", "/about", "/mathematics"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
        health = self.client.get("/health")
        self.assertEqual(health.get_json()["status"], "ok")
        self.assertEqual(health.headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("Content-Security-Policy-Report-Only", health.headers)


if __name__ == "__main__":
    unittest.main()
