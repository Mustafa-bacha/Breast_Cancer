document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect form data
    const formData = {};
    const inputs = document.querySelectorAll('#prediction-form input');
    inputs.forEach(input => {
        formData[input.name] = parseFloat(input.value);
    });

    // Send data to the API
    try {
        const response = await fetch('/predict_cancer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        // Display the result
        const resultDiv = document.getElementById('result');
        resultDiv.classList.remove('hidden');

        document.getElementById('result-message').textContent = result.result;
        document.getElementById('prediction').textContent = result.prediction;
        document.getElementById('probability').textContent = `${(result.probability_benign * 100).toFixed(2)}%`;
        document.getElementById('confidence').textContent = result.confidence_level;
        document.getElementById('confidence-explanation').textContent = result.confidence_explanation;
        document.getElementById('timestamp').textContent = result.timestamp;

        // Scroll to the result
        resultDiv.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        alert('An error occurred while making the prediction. Please try again.');
        console.error(error);
    }
});