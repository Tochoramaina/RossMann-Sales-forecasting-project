import mlflow 
import tensorflow as tf 
import os 

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rossmann-deep-learning")

MODEL_FILE = ""

def register_existing_pretrained_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f'Saved model not in {MODEL_FILE}')

    model = tf.keras.models.load_model(MODEL_FILE)
    with mlflow.start_run(run_name='pretrained_dl_model_import'):
        mlflow.log_param('architecture', 'Hybrid model (CNN & LSTM)')
        mlflow.log_param("pre_trained", True)

        model_info = mlflow.tensorflow.log_model(model = model, artifact_path='model', registered_model_name="RossmannDeepLearningModel")
        print(f'Model logged with URI: {model_info.model_uri}')


if __name__ == "__main__":
    register_existing_pretrained_model()