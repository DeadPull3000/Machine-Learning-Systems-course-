/**
 * app.js - Main Application Bootstrap
 * 
 * Hidden Technical Debt in Machine Learning Systems (Sculley et al., NeurIPS 2015)
 * Interactive Academic Systems Simulation
 */

import { DependencyGraph } from './graph.js';
import { SimulationState } from './state.js';
import { GraphRenderer } from './renderer.js';
import { UIManager } from './ui.js';

document.addEventListener('DOMContentLoaded', () => {
  try {
    const svgElement = document.getElementById('architecture-canvas');
    if (!svgElement) {
      console.error("Failed to find #architecture-canvas");
      return;
    }

    const graph = new DependencyGraph();
    const state = new SimulationState(graph);
    const renderer = new GraphRenderer(svgElement, graph, state);
    const ui = new UIManager(state, renderer);

    // Initial render
    renderer.render();

    console.info("ML Technical Debt Simulation initialized successfully.");
  } catch (err) {
    console.error("Error during simulation initialization:", err);
  }
});
