import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUT_TENSOR = tf.tensor2d(INPUTS);
const OUTPUT_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

let model = tf.sequential();

model.add(
  tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
);
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();

init();

async function init() {
  const exist = Object.keys(await tf.io.listModels());

  if (!exist.includes("localstorage://demo/digitClassifier")) {
    console.log("train");
    train();
  } else {
    console.log("NO train");
    model = await tf.loadLayersModel("localstorage://demo/digitClassifier");
    evaluate();
  }
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

function evaluate() {
  const OFFSET = Math.floor(Math.random() * INPUTS.length);

  let answer = tf.tidy(() => {
    let newInput = tf.tensor1d(INPUTS[OFFSET]);

    let output = model.predict(newInput.expandDims());
    output.print();

    return output.squeeze().argMax();
  });

  answer.array().then((index) => {
    PREDICTION_ELEMENT.innerText = index;
    PREDICTION_ELEMENT.setAttribute(
      "class",
      index === OUTPUTS[OFFSET] ? "correct" : "wrong"
    );
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

async function train() {
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  let results = await model.fit(INPUT_TENSOR, OUTPUT_TENSOR, {
    shuffle: true,
    validationSplit: 0.2,
    batchSize: 512,
    epochs: 50,
  });

  OUTPUT_TENSOR.dispose();
  INPUT_TENSOR.dispose();

  await model.save("localstorage://demo/digitClassifier");

  evaluate();
}

//DRAW THE IMAGE IN THE CANVAS

const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d");

const drawImage = (digit) => {
  var imageData = CTX.getImageData(0, 0, 28, 28);

  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255;
    imageData.data[i * 4 + 1] = digit[i] * 255;
    imageData.data[i * 4 + 2] = digit[i] * 255;
    imageData.data[i * 4 + 3] = 255;
  }

  CTX.putImageData(imageData, 0, 0);

  setTimeout(evaluate, 2000);
};
