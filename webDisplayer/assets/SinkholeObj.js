Sinkhole = function(i, j, size=20) {
    this.i = i;
    this.j = j;

    this.size = size;

    this.x = this.i * (gridWidth / cols) + (gridWidth / cols) / 2;
    this.y = this.j * (gridHeight / rows) + (gridHeight / rows) / 2;

    this.demand = {};

    this.addDemand = function(material, amount) {
        if (!this.demand[material]) {
            this.demand[material] = 0;
        }
        this.demand[material] += amount;
    };

    this.getVertices = function() {
        let size = this.size;
        return [
            { x: this.x, y: this.y - size },
            { x: this.x - size, y: this.y },
            { x: this.x, y: this.y + size },
            { x: this.x + size, y: this.y }
        ];
    }
    // Draw a red diamond to represent the sinkhole on the canvas
    this.show = function() {
        fill(255, 0, 0);
        stroke(0);
        beginShape();
        let vertices = this.getVertices();
        for (let v of vertices) {
            vertex(v.x, v.y);
        }
        endShape(CLOSE);
    }

    this.isClicked = function(mx, my) {
		let x = this.x;
		let y = this.y;
		return dist(mx, my, x, y) < this.size / 2;
	};

    this.logSink = function() {
        console.log("Sink at (" + this.i + ", " + this.j + ")");
        console.log("Demand")
        for (let material in this.demand) {
            console.log("  " + material + ": " + this.demand[material]);
        }
    }
}