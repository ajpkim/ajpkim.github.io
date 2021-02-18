/*
Interactive Boids Simulation
Author: Alex Kim (ajpkim)
Date: 2021-01-20
Repo: https://github.com/ajpkim/interactive-boids
*/

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

canvas.width = canvas.clientWidth;
canvas.height = canvas.width * 0.7;

const MAX_X = canvas.width;
const MAX_Y = canvas.height;

//////////////////////////////////////////////////
// Sliders
const alignRangeSlider = document.getElementById('alignRangeSlider');
const alignForceSlider = document.getElementById('alignForceSlider');
const separateRangeSlider = document.getElementById('separateRangeSlider');
const separateForceSlider = document.getElementById('separateForceSlider');
const cohereRangeSlider = document.getElementById('cohereRangeSlider');
const cohereForceSlider = document.getElementById('cohereForceSlider');
const predatorAvoidanceRangeSlider = document.getElementById('predatorAvoidanceRangeSlider');
const predatorAvoidanceForceSlider = document.getElementById('predatorAvoidanceForceSlider');
const predatorPerceptionRangeSlider = document.getElementById('predatorPerceptionRangeSlider');
const predatorCountSlider = document.getElementById('predatorCountSlider');
//////////////////////////////////////////////////
// Slider variables
let alignRange;
let separateRange;
let cohereRange;
let predatorAvoidanceRange;

let alignMagnitude;
let separateMagnitude;
let cohereMagnitude;
let predatorAvoidanceMagnitude;
//////////////////////////////////////////////////
// Simulation variables
// Boids
let boidMinSpeed = 2;
let boidMaxSpeed = 3;
let boidInitTailLength = 5;
let boidInitSideLength = 9;
let boidInitSideRadians = 2.5;
let boidInitColor = 'blue';
// Predators
let predatorMinSpeed = 1;
let predatorExplodeSpeed = 0.05;
let predatorInitChaseSpeed = boidMaxSpeed * 1.4;
let predatorInitTailLength = boidInitTailLength * 2.25;
let predatorInitSideLength = boidInitSideLength * 1.85;
let predatorInitSideRadians = 2.75;
let predatorCatchRadius = 8;
let predatorExplodeThreshold = 10;
let predatorExplodeCountdown = 120;
let predatorInitColor = 'red';

// General
const rangeMultiplier = 10;
const forceMultiplier = 0.04;
let edgeBuffer = 20;
let numBoids = 250;
let boidIdCount = 0;
//////////////////////////////////////////////////
// Helper functions:

// Return a random number within range [min, max].
function randomNumber(min, max) {
    return Math.random() * (max - min) + min;
}

// Make boid position wrap around canvas instead of fly continually off-screen.
function wrapEdges(boid) {
    if (boid.position.x + edgeBuffer < 0) {
	boid.position.x = MAX_X;
    } else if (boid.position.x - edgeBuffer > MAX_X) {
	boid.position.x = 0;
    }
    if (boid.position.y + edgeBuffer < 0) {
	boid.position.y = MAX_Y
    } else if (boid.position.y - edgeBuffer > MAX_Y) {
	boid.position.y = 0;
    }
}

//////////////////////////////////////////////////
class Vector2d {
    constructor(x, y) {
	this.x = x;
	this.y = y;
    };

    copy() {
	return new Vector2d(this.x, this.y);
    }

    magnitude() {
	return Math.sqrt(this.x**2 + this.y**2);
    }

    // Compute unit vector in current direction and scale to desired magnitude.
    setMagnitude(mag) {
	if (this.magnitude() === 0) {
	    return this;
	} else {
	    this.div(this.magnitude());
	    this.mult(mag);
	    return this;
	}
    }

    limitMagnitude(limit) {
	if (this.magnitude() > limit) {
	    this.setMagnitude(limit);
	}
	return this;
    }

    normalize() {
	return this.div(this.magnitude());
    }

    euclideanDistance(other) {
	return Math.sqrt((this.x - other.x)**2 + (this.y - other.y)**2);
    }

    // Basic vector and scalar arithmetic operations:
    add(other) {
	if (other instanceof Vector2d) {
	    this.x += other.x;
	    this.y += other.y;
	    return this;
	} else if (typeof other === "number") {
	    this.x += other;
	    this.y += other;
	    return this;
	} else {
	    throw new TypeError(`Invalid addition. ${other} is neither vector nor scalar`);
	}
    }

    subtract(other) {
	if (other instanceof Vector2d) {
	    this.x -= other.x;
	    this.y -= other.y;
	    return this;
	} else if (typeof other === "number") {
	    this.x -= other;
	    this.y -= other;
	    return this;
	} else {
	    throw new TypeError(`Invalid subtraction. ${other} is neither vector nor scalar`);
	}
    }

    // Scalar multipication.
    mult(other) {
	if (typeof other === "number") {
	    this.x *= other;
	    this.y *= other;
	    return this;
	} else {
	    throw new TypeError(`Invalid scalar multiplication. ${other} is not a scalar`);
	}
    }

    // Scalar division.
    div(other) {
	if (other === 0) {
	    throw new Error(`Zero Division Error`);
	} else if (typeof other === "number") {
	    this.x /= other;
	    this.y /= other;
	    return this;
	} else {
	    throw new TypeError(`Invalid scalar division. ${other} is not a scalar`);
	}
    }

    dot(other) {
	if (other instanceof Vector2d) {
	    return (this.x * other.x) + (this.y * other.y);
	} else {
	    throw new TypeError(`Invalid dot product. ${other} is not a Vector2d instance`);
	}
    }

    // Calculate and return the angle between this vector and arg vector.
    angleBetween(other) {
	if (other instanceof Vector2d) {
	    let cosineTheta = this.dot(other) / (this.magnitude() + other.magnitude());
	    return Math.acos(cosineTheta);
	} else {
	    throw new TypeError(`Cannot compute angle between ${this} and arg: ${other}`);
	}
    }

    toString() {
	return `<Vector2d obj with position (${this.x, this.y})>`;
    }
}
//////////////////////////////////////////////////
class Boid {
    constructor() {
	this.id = boidIdCount++;
	this.flockIndex = null;
	// Position and Speed
	this.position = new Vector2d(randomNumber(0, MAX_X), randomNumber(0, MAX_Y));
	this.velocity = new Vector2d(randomNumber(-boidMaxSpeed, boidMaxSpeed), randomNumber(-boidMaxSpeed, boidMaxSpeed))
	this.velocity.setMagnitude(randomNumber(boidMinSpeed, boidMaxSpeed));
	this.acceleration = new Vector2d(0, 0);
	// Measurements
	this.sideRadians = boidInitSideRadians;
	this.sideLength = boidInitSideLength;
	this.tailLength = boidInitTailLength;
	this.color = boidInitColor;
	// Flockmates that impact steering
	this.alignMates = [];
	this.separateMates = [];
	this.cohereMates = [];
    }

    // Velocity matching: steer towards the average velocity (orientation) of local flockmates.
    align() {
	if (this.alignMates.length === 0) return;
	let alignForce = new Vector2d(0, 0);
	for (let other of this.alignMates) {
	    alignForce.add(other.velocity);
	}
	alignForce.div(this.alignMates.length);
	alignForce.setMagnitude(this.velocity.magnitude());
	alignForce.subtract(this.velocity);
	alignForce.setMagnitude(alignMagnitude);
	this.acceleration.add(alignForce);
    }
    // Collision avoidance: steer to avoid crowding local flockmates.
    separate() {
	if (this.separateMates.length === 0) return;
	let separateForce = new Vector2d(0, 0);
	for (let other of this.separateMates) {
	    let distance = boidDistances[this.flockIndex][other.flockIndex];
	    if (distance === 0) distance = 0.01;
	    let diffVector = this.position.copy().subtract(other.position);
	    // Weight further away boids less.
	    diffVector.div(distance**2);
	    separateForce.add(diffVector);
	}
	separateForce.setMagnitude(this.velocity.magnitude());
	separateForce.subtract(this.velocity);
	separateForce.setMagnitude(separateMagnitude);
	this.acceleration.add(separateForce);
    }
    // Flock centering: steer towards the average position of local flockmates.
    cohere() {
	if (this.cohereMates.length === 0) return;
	let cohereForce = new Vector2d(0, 0);
	for (let other of this.cohereMates) {
	    cohereForce.add(other.position);
	};
	cohereForce.div(this.cohereMates.length);
	// Get vector pointing towards average location by subtracting current location.
	cohereForce.subtract(this.position);
	cohereForce.subtract(this.velocity);
	cohereForce.setMagnitude(cohereMagnitude);
	this.acceleration.add(cohereForce);
    }

    avoidPredators() {
	let predatorAvoidanceForce = new Vector2d(0,0);
	for (let predator of predators) {
	    let distance = this.position.euclideanDistance(predator.position);
	    if (distance <= predatorAvoidanceRange) {
		let diffVector = this.position.copy().subtract(predator.position);
		diffVector.div(distance ** 2);
		predatorAvoidanceForce.add(diffVector);
	    }
	}
	predatorAvoidanceForce.setMagnitude(predatorAvoidanceMagnitude);
	this.acceleration.add(predatorAvoidanceForce);
    }

    // Calculate the align, separate, and cohere forces and apply them to acceleration.
    applySteeringForces() {
	this.align();
	this.cohere();
	this.separate();
	this.avoidPredators();
    }

    updatePosition() {
	this.applySteeringForces()
	this.velocity.add(this.acceleration);
	this.velocity.setMagnitude(Math.max(this.velocity.magnitude(), boidMinSpeed));
	this.velocity.limitMagnitude(boidMaxSpeed);
	this.position.add(this.velocity);
	wrapEdges(this);
	this.acceleration.mult(0);
    }

    getDrawingCoords() {
	let points = [];
	let directionVector = this.velocity.copy().normalize();
	let directionRadians = Math.atan2(directionVector.y, directionVector.x);

	let side1Radians = directionRadians + this.sideRadians;
	let side1 = new Vector2d(Math.cos(side1Radians), Math.sin(side1Radians));
	side1.setMagnitude(this.sideLength);
	side1 = this.position.copy().add(side1);
	points.push([side1.x, side1.y]);

	let tailVector = directionVector.mult(-1).setMagnitude(this.tailLength);
	let tailPoint = this.position.copy().add(tailVector);
	points.push([tailPoint.x, tailPoint.y]);

	let side2Radians = directionRadians - this.sideRadians;
	let side2 = new Vector2d(Math.cos(side2Radians), Math.sin(side2Radians));
	side2.setMagnitude(this.sideLength);
	side2 = this.position.copy().add(side2);
	points.push([side2.x, side2.y]);

	// Add this head position as final coordinate for drawing loop.
	points.push([this.position.x, this.position.y]);

    return points
    }

}

class Predator {
    constructor () {
	// Position and Speed
	this.position = new Vector2d(randomNumber(0, MAX_X), randomNumber(0, MAX_Y));
	this.velocity = new Vector2d(randomNumber(-predatorInitChaseSpeed, predatorInitChaseSpeed), randomNumber(-predatorInitChaseSpeed, predatorInitChaseSpeed))
	this.velocity.setMagnitude(randomNumber(predatorMinSpeed, predatorInitChaseSpeed));
	this.acceleration = new Vector2d(0, 0);
	this.chaseSpeed = predatorInitChaseSpeed;
	// Measurements
	this.sideRadians = predatorInitSideRadians
	this.sideLength = predatorInitSideLength;
	this.tailLength = predatorInitTailLength;
	this.color = predatorInitColor;
	// Status
	this.prey = null;
	this.preyEatCount = 0;
	this.explodeCountdown = null;
    }

    // Loop through flock until find a boid within perception range, lock that boid as "prey".
    findPrey() {
	let prey;
	for (let boid of flock) {
	    if (this.position.euclideanDistance(boid.position) <= predatorPerceptionRange) {
		this.prey = boid;
		return
	    }
	}
	this.prey = null;
    }

    eatPrey() {
	let preyIndex = flock.indexOf(this.prey);
	flock.splice(preyIndex, 1);
	this.prey = null;
	this.preyEatCount += 1;

	if (this.preyEatCount < predatorExplodeThreshold) {
	    this.sideRadians -= 0.05
	    this.tailLength *= 1.15;
	    this.chaseSpeed *= 0.95;
	} else {
	    this.explodeCountdown = predatorExplodeCountdown;
	    this.velocity.setMagnitude(predatorExplodeSpeed);
	}
    }

    explode() {
	// Remove this predator and create new one.
	let selfIndex = predators.indexOf(this);
	predators.splice(selfIndex, 1);
	predators.push(new Predator());
	// Create new boids in flock at current position.
	for (let i = 0; i < this.preyEatCount; i++) {
	    let boid = new Boid();
	    // Adding random number to avoid Zero Division Error in distance calucaltions.
	    boid.position = this.position.copy().add(randomNumber(-3, 3));
	    flock.push(boid);
	}
    }

    explodeStep() {
	this.tailLength *= 1.005;
	this.sideLength *= 1.005;
	this.explodeCountdown -= 1
	if (this.explodeCountdown > 100) {
	    this.color = 'black';
	} else if (this.explodeCountdown > 50) {
	    this.color = 'purple';
	} else if (this.explodeCountdown > 25) {
	    this.color = 'blue';
	} else {
	    this.explode()
	}
    }

    chasePrey() {
	if ((this.prey === null) || (this.position.euclideanDistance(this.prey.position) > predatorPerceptionRange)) {
	    this.findPrey();
	}
	if (this.prey === null) {
	    // Apply 20% braking force acceleration
	    let brakingForce = this.velocity.copy().mult(-1);
	    brakingForce.setMagnitude(this.velocity.magnitude() * 0.2);
	    this.acceleration.add(brakingForce);
	} else if (this.position.euclideanDistance(this.prey.position) < predatorCatchRadius) {
	    this.eatPrey();
	} else {
	    let chaseForce = this.prey.position.copy().subtract(this.position);
	    chaseForce.subtract(this.velocity);
	    chaseForce.setMagnitude(this.chaseSpeed);
	    this.acceleration.add(chaseForce);
	}
    }

    updatePosition() {
	if (typeof this.explodeCountdown === "number") {
	    this.explodeStep();
	} else {
	    this.chasePrey();
	    this.velocity.add(this.acceleration);
	    this.velocity.setMagnitude(Math.max(predatorMinSpeed, this.velocity.magnitude()));
	    if (this.prey != null) this.velocity.setMagnitude(this.chaseSpeed);
	}
	this.position.add(this.velocity);
	this.acceleration.mult(0);
	wrapEdges(this);
    }

    getDrawingCoords() {
	let points = [];
	let directionVector = this.velocity.copy().normalize();
	let directionRadians = Math.atan2(directionVector.y, directionVector.x);

	let side1Radians = directionRadians + this.sideRadians;
	let side1 = new Vector2d(Math.cos(side1Radians), Math.sin(side1Radians));
	side1.setMagnitude(this.sideLength);
	side1 = this.position.copy().add(side1);
	points.push([side1.x, side1.y]);

	let tailVector = directionVector.mult(-1).setMagnitude(this.tailLength);
	let tailPoint = this.position.copy().add(tailVector);
	points.push([tailPoint.x, tailPoint.y]);

	let side2Radians = directionRadians - this.sideRadians;
	let side2 = new Vector2d(Math.cos(side2Radians), Math.sin(side2Radians));
	side2.setMagnitude(this.sideLength);
	side2 = this.position.copy().add(side2);
	points.push([side2.x, side2.y]);

	// Add this head position as final coordinate for drawing loop.
	points.push([this.position.x, this.position.y]);
	return points
    }

}
//////////////////////////////////////////////////
// Calculate and store the distance between each boid in a
// global 2d matrix and update each boids flockmates array.
function calcDistancesFlockmates() {
    for (let [i, boid] of flock.entries()) {
	boid.flockIndex = i;
	boid.alignMates.length = 0;
	boid.separateMates.length = 0;
	boid.cohereMates.length = 0;
	// Only iterate through each pair one time.
	for (let j = i + 1; j < flock.length; j++) {
	    let other = flock[j]
	    other.flockIndex = j
	    let distance = boid.position.euclideanDistance(other.position);
	    // Update global distance matrix.
	    boidDistances[i][j] = distance;
	    boidDistances[j][i] = distance;
	    // Populate the flockmates for each force.
	    if (distance <= alignRange) {
		boid.alignMates.push(other);
		other.alignMates.push(boid);
	    }
	    if (distance <= separateRange) {
		boid.separateMates.push(other);
		other.separateMates.push(boid);
	    }
	    if (distance <= cohereRange) {
		boid.cohereMates.push(other);
		other.cohereMates.push(boid);
	    }
	}
    }
}
//////////////////////////////////////////////////
function updatePredators() {
    while (predators.length < predatorPopulation) {
	predators.push(new Predator());
    }
    while (predators.length > predatorPopulation) {
	predators.pop();
    }
    predators.forEach(predator => {
	predator.updatePosition();
	drawElement(predator);
    });
}
//////////////////////////////////////////////////
// Animation and Sliders
function drawElement(element) {
    coordsArr = element.getDrawingCoords();
    ctx.moveTo(element.position.x, element.position.y);
    ctx.beginPath();
    coordsArr.forEach(coords => ctx.lineTo(coords[0], coords[1]));
    ctx.fillStyle = element.color;
    ctx.fill();
}

function updateSliders() {
    // Primary flocking interactions
    alignRange = alignRangeSlider.value * rangeMultiplier;
    separateRange = separateRangeSlider.value * rangeMultiplier;
    cohereRange = cohereRangeSlider.value * rangeMultiplier;

    alignMagnitude = alignForceSlider.value * forceMultiplier;
    separateMagnitude = separateForceSlider.value * forceMultiplier;
    cohereMagnitude = cohereForceSlider.value * forceMultiplier;

    // Predator interactions (discounts on multipliers are arbitrary).
    predatorAvoidanceRange = predatorAvoidanceRangeSlider.value * (rangeMultiplier * 0.6);
    predatorAvoidanceMagnitude = predatorAvoidanceForceSlider.value * (forceMultiplier * 2);
    predatorPerceptionRange = predatorPerceptionRangeSlider.value * (rangeMultiplier * 0.6);
    predatorPopulation = predatorCountSlider.value;
}

function runSim() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    updateSliders();
    updatePredators();
    calcDistancesFlockmates();
    flock.forEach((boid) => {
	drawElement(boid);
	boid.updatePosition();
    });
    requestAnimationFrame(runSim);
}

// For testing.
function step() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    updateSliders();
    calcDistancesFlockmates();
    flock.forEach((boid) => {
	drawElement(boid);
	boid.updatePosition();
    });
}
//////////////////////////////////////////////////
// Initialize and run simulation.
const flock = []
const predators = [];
for (let i=0; i < numBoids; i++) {
    flock.push(new Boid());
}
 // Create 2d array to hold boid distances i.e. [0][2] holds dist from boid 0 to boid 2
let boidDistances = Array.from(Array(flock.length), () => new Array(flock.length))
runSim(flock);
