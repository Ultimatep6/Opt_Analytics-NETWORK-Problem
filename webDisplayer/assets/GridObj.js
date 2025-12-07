function Grid() {
    this.nodes = {};
    this.connections = {};
    
    this.factories = {};
    this.sources = {};
    this.sinks = {};

    this.flowColors = {};

    this.addNode = function(node) {
        this.nodes[`(${node.i},${node.j})`] = node;
    }

    this.addConnection = function(connection) {
        this.connections[`(${connection.a.i},${connection.a.j})-(${connection.b.i},${connection.b.j})`] = connection;
    }

    this.addFactory = function(factory) {
        this.factories[`(${factory.i},${factory.j})`] = factory;
    }

    this.addSource = function(source) {
        this.sources[`(${source.i},${source.j})`] = source;
    }

    this.addSink = function(sink) {
        this.sinks[`(${sink.i},${sink.j})`] = sink;
    }

    this.addMaterialColor = function(matFlow, colorVal) {
        this.flowColors[matFlow] = colorVal;
    }

    this.showDefault = function(connection) {
		stroke(0);
        line(connection.a.x, connection.a.y, connection.b.x, connection.b.y);
    }

	// Draw an arrowed line to represent flow on the connection
	this.showFlow = function(connection, matFlow) {
        if (matFlow === 'default') {
            this.showDefault(connection);
            return;
        }

		this.hideConnection(connection);
		stroke(this.flowColors[matFlow]);
        strokeWeight(2);
		line(connection.a.x, connection.a.y, connection.b.x, connection.b.y);

		// Draw arrowhead
		let angle = atan2(connection.b.y - connection.a.y, connection.b.x - connection.a.x);
		push();
		translate(connection.b.x, connection.b.y);
		rotate(angle);
		let arrowSize = 5;
		line(0, 0, -arrowSize, arrowSize / 2);
		line(0, 0, -arrowSize, -arrowSize / 2);
		pop();
	}

	this.hideConnection = function(connection, matFlow='default') {
		if (matFlow === 'default') {
			stroke(255);
			line(connection.a.x, connection.a.y, connection.b.x, connection.b.y);

		} else if (matFlow !== 'default') {
			// Remove the arrowhead by overdrawing it
			let angle = atan2(connection.b.y - connection.a.y, connection.b.x - connection.a.x);
			push();
			translate(connection.b.x, connection.b.y);
			rotate(angle);
			let arrowSize = 5;
			line(0, 0, -arrowSize, arrowSize / 2);
			line(0, 0, -arrowSize, -arrowSize / 2);
			pop();
		}
	}
}