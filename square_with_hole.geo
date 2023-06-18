SetFactory("OpenCASCADE");
Rectangle(1) = {0,0,0, 1,1};
Rectangle(2) = {.4,.4,0, .2,.2};

Curve Loop(3) = {4, 1, 2, 3};
Curve Loop(4) = {7, 8, 5, 6};
Plane Surface(3) = {3, 4};

// for mesh movement we need a separate boundary id for each straight part of the boundary
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Line(5) = {5};
Physical Line(6) = {6};
Physical Line(7) = {7};
Physical Line(8) = {8};
Physical Surface(1) = {3};

Mesh.MeshSizeMin = .02;
Mesh.MeshSizeMax = .02;
