Factory = function(i, j, gridObj, size=20) {
    this.i = i;
    this.j = j;

    this.x = this.i * (gridWidth / cols) + (gridWidth / cols) / 2;
    this.y = this.j * (gridHeight / rows) + (gridHeight / rows) / 2;

    this.size = size;

    this.demand = {};
    this.supply = {};

    this.gridObj = gridObj;

    this.addDemand = function(material, amount) {
        if (!this.demand[material]) {
            this.demand[material] = 0;
        }
        this.demand[material] += amount;
    };

    this.addSupply = function(material, amount) {
        if (!this.supply[material]) {
            this.supply[material] = 0;
        }
        this.supply[material] += amount;
    };

    this.generateTrianglePoints = function() {
        let size = this.size;
        return [
            { x: this.x, y: this.y - size },
            { x: this.x - size * Math.sin(Math.PI/3), y: this.y + size * Math.cos(Math.PI/3) },
            { x: this.x + size * Math.sin(Math.PI/3), y: this.y + size * Math.cos(Math.PI/3) }
        ];
    }

    // Draw a yellow triangle to represent the factory on the canvas
    this.show = function() {
		fill(255, 204, 0);
		stroke(0);
        let points = this.generateTrianglePoints();
        triangle(points[0].x, points[0].y, points[1].x, points[1].y, points[2].x, points[2].y);
	};

    this.isClicked = function(mx, my) {
		let x = this.x;
		let y = this.y;
		return dist(mx, my, x, y) < this.size / 2;
	};

    this.logFactory = function() {
        console.log("Factory at (" + this.i + ", " + this.j + ")");
        console.log("Demand")
        for (let material in this.demand) {
            console.log(" " + material + ": " + this.demand[material]);
        }
        console.log("Supply")
        for (let material in this.supply) {
            console.log(" " + material + ": " + this.supply[material]);
        }
    }

    this.getDemandInfo = function() {
        return this.demand;
    }

    this.getSupplyInfo = function() {
        return this.supply;
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
