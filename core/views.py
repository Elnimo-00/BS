import pandas as pd
import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from .forms import UploadCSVForm
from .models import UploadedCSV
from django.conf import settings
from .rag_agent import ask_about_data

# Load TensorFlow model (.h5)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'pin_digit_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

def upload_and_analyze(request):
    prediction = None
    telemetry_stats = None
    chatbot_response = None

    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            df = pd.read_csv(uploaded_file.file.path)

            try:
                # Ensure input shape matches what the model expects
                input_data = df.to_numpy()
                input_data = np.expand_dims(input_data, axis=0)  # Shape (1, timesteps, features) if needed
                prediction_raw = model.predict(input_data)
                prediction = int(np.argmax(prediction_raw))  # Assuming classification
            except Exception as e:
                prediction = f"Prediction failed: {e}"

            telemetry_stats = df.describe().to_html()
            user_question = request.POST.get('user_question', '')
            if user_question:
                chatbot_response = ask_about_data(user_question, df)
    else:
        form = UploadCSVForm()

    return render(request, 'core/upload.html', {
        'form': form,
        'prediction': prediction,
        'telemetry_stats': telemetry_stats,
        'chatbot_response': chatbot_response,
    })
