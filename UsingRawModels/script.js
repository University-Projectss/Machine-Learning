const MODEL_PATH =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json";

let model = undefined;

//save the model offline in localstorage
//await model.save('localstorage://demo/newModelName');

//check if there's a model in localstorage already
// console.log(JSON.stringify(await tf.io.listModel()));

//load the local model
//model = await tf.loadLayersModel(MODEL_PATH);

const loadModel = async () => {
  model = await tf.loadLayersModel("localstorage://demo/newModelName");
  model.summary();

  //batch of 1
  const input = tf.tensor2d([[870]]);

  //batch of 3
  const inputBatch = tf.tensor2d([[500], [1100], [970]]);

  const result = model.predict(input);
  const resultBatch = model.predict(inputBatch);

  result.print();
  resultBatch.print();

  input.dispose();
  inputBatch.dispose();
  result.dispose();
  resultBatch.dispose();
  model.dispose();
};

loadModel();
