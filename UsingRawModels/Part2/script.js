const MODEL_PATH =
  "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";
const EXAMPLE_IMG = document.getElementById("exampleImg");
const IMAGE_BOX = document.getElementById("imageBox");
let movenet = undefined;

const loadAndRunModel = async () => {
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });

  //   let exampleTensorInput = tf.zeros([1, 192, 192, 3], "int32");
  let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
  console.log(imageTensor.shape);

  let croppStartPoint = [15, 170, 0]; //y, x, color channel
  let croppSize = [345, 345, 3]; //height, width, color channel

  let croppedTensor = tf.slice(imageTensor, croppStartPoint, croppSize);
  let resizedTensor = tf.image
    .resizeBilinear(croppedTensor, [192, 192], true)
    .toInt();
  console.log(resizedTensor.shape);

  let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
  let arrayOutput = await tensorOutput.array();

  console.log(arrayOutput[0][0]);

  for (let point of arrayOutput[0][0]) {
    const highlighter = document.createElement("div");
    highlighter.setAttribute("class", "highlighter");
    highlighter.style =
      "left: " +
      (point[1] * 345 + 170) +
      "px; top: " +
      (point[0] * 345 + 15) +
      "px; width: " +
      5 +
      "px; height: " +
      5 +
      "px;";

    IMAGE_BOX.appendChild(highlighter);
  }

  imageTensor.dispose();
  croppedTensor.dispose();
  resizedTensor.dispose();
  tensorOutput.dispose();
};

loadAndRunModel();
