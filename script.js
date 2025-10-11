let model;

async function loadModel() {
  // Load the model
  model = await tf.loadLayersModel('tfjs_model/model.json');
  console.log("✅ Model loaded successfully");
}

loadModel();

document.getElementById("file-input").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    document.getElementById("preview").src = URL.createObjectURL(file);
  }
});

async function predict() {
  const image = document.getElementById("preview");
  if (!image.src) {
    alert("Please upload an image first.");
    return;
  }

  // Adjust input size to match your model (e.g., 128x128)
  const tensor = tf.browser.fromPixels(image)
    .resizeNearestNeighbor([128, 128])
    .toFloat()
    .expandDims(0);

  const prediction = await model.predict(tensor);
  const classIndex = prediction.argMax(-1).dataSync()[0];

  // Replace these with your actual class names
  const classNames = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"];

  document.getElementById("result").innerText = 
    `Predicted Class: ${classNames[classIndex] || classIndex}`;
}

