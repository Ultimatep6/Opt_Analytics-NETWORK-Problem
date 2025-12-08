function infoContainer(config = {}) {
    this.container = null; // Store a reference to the container

    // Default configuration
    const defaultConfig = {
        x: 0, // Default x-position
        y: 0, // Default y-position
        width: '100%', // Default width
        height: 'auto', // Default height
        border: '1px solid black', // Default border
        padding: '10px', // Default padding
        marginTop: '10px', // Default margin-top
    };

    // Merge user-provided config with default config
    this.config = { ...defaultConfig, ...config };

    this.setup = function() {
        // Create a container div
        this.container = createDiv();
        this.container.id('infoContainer'); // Assign an ID for easier reference

        // Apply styles based on the configuration
        this.container.style('position', 'absolute'); // Use absolute positioning for custom placement
        this.container.style('left', `${this.config.x}px`);
        this.container.style('top', `${this.config.y}px`);
        this.container.style('width', this.config.width);
        this.container.style('height', this.config.height);
        this.container.style('border', this.config.border);
        this.container.style('padding', this.config.padding);
        this.container.style('margin-top', this.config.marginTop);
        this.container.style('box-sizing', 'border-box'); // Include padding in width/height

        // Add some initial content to the container
        this.title_container = createDiv();
        this.title_container.parent(this.container);
        this.title_container.style('text-align', 'center');
        let title = createElement('h2', 'Special Node Information');
        title.parent(this.title_container); // Add the title to the title container

        this.paragraph_container = createDiv();
        this.paragraph_container.parent(this.container);
        this.paragraph_container.style('text-align', 'center');
        let paragraph = createP('Click on a factory, sinkhole, or source to see details.');
        paragraph.parent(this.paragraph_container); // Add the paragraph to the paragraph container
        
        this.supp_dem_container = createDiv();
        this.supp_dem_container.parent(this.container);
        this.supp_dem_container.style('display', 'flex');
        this.supp_dem_container.style('justify-content', 'space-between');
        this.supp_dem_container.style('gap', '24px'); // Add gap between containers


        this.demand_container = createDiv();
        this.demand_container.parent(this.supp_dem_container);
        this.demand_container.style('text-align', 'left');
        this.demand_container.style('min-width', '120px'); // Set a fixed or min width
        this.demand_container.style('flex', '1');

        this.supply_container = createDiv();
        this.supply_container.parent(this.supp_dem_container);
        this.supply_container.style('text-align', 'left');
        this.supply_container.style('min-width', '120px'); // Set a fixed or min width
        this.supply_container.style('flex', '1');

    };

    this.updateContent = function(newContent) {
        // Clear the container and add new content
        this.paragraph_container.html(''); // Clear existing content
        let paragraph = createP(newContent);
        paragraph.parent(this.paragraph_container);
    };  

    this.clearDemand = function() {
        this.demand_container.html('');
    }

    this.clearSupply = function() {
        this.supply_container.html('');
    }

    // Updates the demand_container with demand info from demandObj
    this.updateDemand = function(demandObj) {
        this.demand_container.html('');
        // Use demandObj.stringifyInfo if available, otherwise fallback to this.getDemandInfo
        let text = typeof demandObj.stringifyInfo === 'function'
            ? demandObj.stringifyInfo(true)
            : this.getDemandInfo.call(demandObj);
        this.demand_container.html(text);
    };

    this.updateSupply = function(supplyObj) {
        this.supply_container.html('');
        // Use supplyObj.stringifyInfo if available, otherwise fallback to this.getSupplyInfo
        let text = typeof supplyObj.stringifyInfo === 'function'
            ? supplyObj.stringifyInfo(false)
            : this.getSupplyInfo.call(supplyObj);
        this.supply_container.html(text);
    };

    // Adjust layout: show both side by side if both have content, otherwise show only the one with content full width
    this.updateLayout = function() {
        const demandText = this.demand_container.html().trim();
        const supplyText = this.supply_container.html().trim();

        if (demandText && supplyText) {
            // Both have content: show side by side
            this.supp_dem_container.style('flex-direction', 'row');
            this.demand_container.style('display', 'block');
            this.supply_container.style('display', 'block');
        } else if (demandText) {
            // Only demand: show full width, hide supply
            this.supp_dem_container.style('flex-direction', 'column');
            this.demand_container.style('display', 'block');
            this.supply_container.style('display', 'none');
        } else if (supplyText) {
            // Only supply: show full width, hide demand
            this.supp_dem_container.style('flex-direction', 'column');
            this.demand_container.style('display', 'none');
            this.supply_container.style('display', 'block');
        } else {
            // Neither: hide both
            this.demand_container.style('display', 'none');
            this.supply_container.style('display', 'none');
        }
    };
};
