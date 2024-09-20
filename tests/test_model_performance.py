import unittest
import tensorflow as tf

class TestModelPerformance(unittest.TestCase):

    def test_model_accuracy(self):
        # Load the best model and validate its performance
        model = tf.keras.models.load_model("models/best_overall_model.keras")

        # Simulate or use actual validation data
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            'data/val',
            labels='inferred',
            label_mode='categorical',
            batch_size=32,
            image_size=(256, 256),
            shuffle=False
        )

        loss, acc, top_k_acc = model.evaluate(val_dataset)

        # Check if the accuracy meets the minimum acceptance criteria
        self.assertGreaterEqual(acc, 0.85)  # e.g., the model should achieve >= 85% accuracy

    def tearDown(self):
        # Clean up any artifacts if needed
        pass

if __name__ == '__main__':
    unittest.main()
