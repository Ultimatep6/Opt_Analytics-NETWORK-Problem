Source = function(i, j, supply, size=20) {
    this.i = i;
    this.j = j;

    this.size = size;

    this.x = this.i * (gridWidth / cols) + (gridWidth / cols) / 2;
    this.y = this.j * (gridHeight / rows) + (gridHeight / rows) / 2;

    this.supply = {};

    this.addSupply = function(material, amount) {
        if (!this.supply[material]) {
            this.supply[material] = 0;
        }
        this.supply[material] += amount;
    };

    // draw a blue hexagon to represent the source on the canvas
    this.show = function() {
        fill(0, 0, 255);
        stroke(0);
        beginShape();
        for (let i = 0; i < 6; i++) {
            let angle = TWO_PI / 6 * i;
            let x = this.x + cos(angle) * this.size;
            let y = this.y + sin(angle) * this.size;
            vertex(x, y);
        }
        endShape(CLOSE);
    }

    this.isClicked = function(mx, my) {
		let x = this.x;
		let y = this.y;
		return dist(mx, my, x, y) < this.size / 2;
	};

    this.logSource = function() {
        console.log("Source at (" + this.i + ", " + this.j + ")");
        console.log("Supply")
        console.log(this.supply)
        for (let material in this.supply) {
            console.log("  " + material + ": " + this.supply[material]);
        }
    }
}
