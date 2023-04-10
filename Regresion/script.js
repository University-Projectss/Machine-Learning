/////////////////// Preprocesing Data ///////////////////

import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

//Shuffle both array in the same way, so the inputs will still corespond to outputs
tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUT_TENSOR = tf.tensor2d(INPUTS);
const OUTPUT_TENSOR = tf.tensor1d(OUTPUTS);

const normalize = (tensor, min, max) => {
  const result = tf.tidy(() => {
    //find the min and max values in the tensor
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);

    //substract the min values from each element from the tensor
    const TENSOR_SUBSTRACT_MIN = tf.sub(tensor, MIN_VALUES);

    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    //actual normalization
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBSTRACT_MIN, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });

  return result;
};

const FEATURE_RESULTS = normalize(INPUT_TENSOR);

INPUT_TENSOR.dispose();

/////////////////// Creating the Model ///////////////////

//A sequential model where the output of a layer becomes
//the input of the next layer
const model = tf.sequential();

//we are adding one neuron (units: 1), that is connected to all
//inputs (inputShape: [2])
model.add(tf.layers.dense({ inputShape: [2], units: 1 }));

model.summary();

const evaluate = () => {
  //predict answer for a single piece of data
  //using tidy to automaticaly dispose the tensors
  tf.tidy(() => {
    let newInput = normalize(
      tf.tensor2d([[750, 1]]),
      FEATURE_RESULTS.MIN_VALUES,
      FEATURE_RESULTS.MAX_VALUES
    );

    let newOutput = model.predict(newInput.NORMALIZED_VALUES);
    newOutput.print();
  });

  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();
};

const train = async () => {
  const LEARNING_RATE = 0.01;

  //compile the model
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError",
  });

  let results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUT_TENSOR,
    {
      validationSplit: 0.15, //take aside 15% of inputs for testing
      suffle: true, //avoid learning from the data order
      batchSize: 64, //the number of examples tried before adjusting weights
      epochs: 10, //go over data 10 times
    }
  );

  OUTPUT_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  console.log(
    "Average error loss: " +
      Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );
  console.log(
    "Average validation error loss: " +
      Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
  );

  await model.save("localstorage://demo/FirstNeuronModel");
  evaluate();
};

train();
