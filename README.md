# VRPOS
This repository contains the source code for my project, focusing on VRPOS (Vehicle Route Planning with Optinal Subtours) and mini-hub system.

The VRPOS modeling aims to simulate the driving, parking, and walking behaviors of couriers. The problem is decomposed into two levels: clustering at the first level and route planning at the second level. The walking subtour is optional, so the second-level problem involves more complex decompositions. The most important aspect, the selection of parking locations, is solved using genetic algorithm.

<h2>Delivery mode</h2>
<img src="./delivery.png" width="500" />

<h2>Definition of VRPOS</h2>
<img src="./vrpos explained.png" width="800" />

<h2>Decomposition</h2>
<img src="./subproblems.png" width="700" />

<h2>Solving VRPOS</h2>
<img src="./solving.png" width="500" />
