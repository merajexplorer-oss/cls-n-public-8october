let model;

// Replace with your actual class names
const classes = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]; 

// Load the TF.js model
async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log('Model loaded');
}

loadModel();

// Handle image upload
document.getElementById('imageInput').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const previewImage = document.getElementById('previewImage');
    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = 'block';

    previewImage.onload = async () => {
        let tensor = tf.browser.fromPixels(previewImage)
            .resizeNearestNeighbor([224, 224]) // replace 224x224 with your model input size
            .toFloat()
            .div(255.0)
            .expandDims();

        const prediction = model.predict(tensor);
        const classIndex = prediction.argMax(-1).dataSync()[0];
        document.getElementById('result').innerText = `Predicted Class: ${classes[classIndex]}`;
    };
});
