const canvas = document.getElementById("canvas");
canvas.width = 200;

const ctx = canvas.getContext("2d");
const road = new Road(canvas.width / 2, canvas.width * 0.9);
const car = new Car(road.getLaneCenter(1), 200, 30, 50);

animate();

function animate() {
  car.update();

  //this will erase the canvas
  canvas.height = window.innerHeight;

  ctx.save();
  ctx.translate(0, canvas.height * 0.8 - car.y);
  road.draw(ctx);
  car.draw(ctx);

  ctx.restore();
  //this calls the function many times to give the animation effect
  requestAnimationFrame(animate);
}
