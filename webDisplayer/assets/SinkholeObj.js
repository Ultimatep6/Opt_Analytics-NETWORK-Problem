Sinkhole = function(i, j, gridObj, size=20) {
    this.i = i;
    this.j = j;

    this.size = size;

    this.x = this.i * (gridWidth / cols) + (gridWidth / cols) / 2;
    this.y = this.j * (gridHeight / rows) + (gridHeight / rows) / 2;

    this.demand = {};

    this.gridObj = gridObj;

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

    this.getDemandInfo = function() {
        return this.demand;
    }

    this.stringifyInfo = function(demand = false) {
        console.log(this.gridObj.flowColors);
        let info = demand ? 'Demand:<br>' : 'Supply:<br>';
        let materials = demand ? this.demand : this.supply;
        for (let material in materials) {
            let value = materials[material];
            let color = "#cccccc"; // default gray
            if (this.gridObj.flowColors && this.gridObj.flowColors[material] && value !== 0) {
                color = this.gridObj.flowColors[material];
            }
            let amount = demand ? value : value * -1;
            info += `<span style="display:inline-block;width:12px;height:12px;background:${color};margin-right:6px;border:1px solid #888;"></span>`;
            info += `${material}: ${amount}<br>`;
        }
        return info;
    }
}