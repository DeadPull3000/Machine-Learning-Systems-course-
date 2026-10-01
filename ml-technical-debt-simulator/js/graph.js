/**
 * graph.js - Internal Dependency Graph Data Model
 * 
 * Implements a directed graph with cycle-safe traversal for simulating
 * ML system dependencies and calculating downstream impact closures (CACE).
 * 
 * References:
 * Sculley et al., "Hidden Technical Debt in Machine Learning Systems", NeurIPS 2015
 * Section 2: Complex Models Lose Boundaries (CACE principle)
 */

export class DependencyGraph {
  constructor() {
    this.nodes = new Map(); // id -> Node
    this.edges = new Map(); // id -> Edge
    this.adjacency = new Map(); // id -> Set of edgeIds (outgoing)
    this.reverseAdjacency = new Map(); // id -> Set of edgeIds (incoming)
  }

  clear() {
    this.nodes.clear();
    this.edges.clear();
    this.adjacency.clear();
    this.reverseAdjacency.clear();
  }

  addNode(node) {
    if (!node.id) throw new Error("Node must have an id");
    const nodeObj = {
      id: node.id,
      label: node.label || node.id,
      sublabel: node.sublabel || '',
      type: node.type || 'generic', // 'dataset', 'etl', 'feature', 'model', 'correction_model', 'prediction', 'consumer', 'user', 'feedback'
      category: node.category || 'core', // 'input', 'processing', 'model', 'serving', 'consumer', 'feedback'
      status: node.status || 'normal', // 'normal', 'highlight', 'affected', 'warning', 'healthy'
      meta: node.meta || {},
      col: node.col ?? 0,
      row: node.row ?? 0
    };
    this.nodes.set(nodeObj.id, nodeObj);
    if (!this.adjacency.has(nodeObj.id)) {
      this.adjacency.set(nodeObj.id, new Set());
    }
    if (!this.reverseAdjacency.has(nodeObj.id)) {
      this.reverseAdjacency.set(nodeObj.id, new Set());
    }
    return nodeObj;
  }

  removeNode(id) {
    if (!this.nodes.has(id)) return;
    
    // Remove outgoing edges
    const outgoing = this.adjacency.get(id) || new Set();
    outgoing.forEach(edgeId => this.removeEdge(edgeId));

    // Remove incoming edges
    const incoming = this.reverseAdjacency.get(id) || new Set();
    incoming.forEach(edgeId => this.removeEdge(edgeId));

    this.adjacency.delete(id);
    this.reverseAdjacency.delete(id);
    this.nodes.delete(id);
  }

  getNode(id) {
    return this.nodes.get(id);
  }

  getAllNodes() {
    return Array.from(this.nodes.values());
  }

  addEdge(edge) {
    const id = edge.id || `${edge.source}->${edge.target}`;
    if (!this.nodes.has(edge.source) || !this.nodes.has(edge.target)) {
      console.warn(`Cannot add edge ${id}: missing source or target node`);
      return null;
    }

    const edgeObj = {
      id,
      source: edge.source,
      target: edge.target,
      label: edge.label || '',
      type: edge.type || 'data_dependency', // 'data_dependency', 'correction', 'consumer_feed', 'feedback_loop'
      animated: !!edge.animated,
      dashed: !!edge.dashed,
      status: edge.status || 'normal', // 'normal', 'affected', 'warning'
      routing: edge.routing || 'direct' // 'direct', 'elbow', 'loop'
    };

    this.edges.set(id, edgeObj);

    if (!this.adjacency.has(edge.source)) this.adjacency.set(edge.source, new Set());
    this.adjacency.get(edge.source).add(id);

    if (!this.reverseAdjacency.has(edge.target)) this.reverseAdjacency.set(edge.target, new Set());
    this.reverseAdjacency.get(edge.target).add(id);

    return edgeObj;
  }

  removeEdge(id) {
    const edge = this.edges.get(id);
    if (!edge) return;
    if (this.adjacency.has(edge.source)) {
      this.adjacency.get(edge.source).delete(id);
    }
    if (this.reverseAdjacency.has(edge.target)) {
      this.reverseAdjacency.get(edge.target).delete(id);
    }
    this.edges.delete(id);
  }

  getEdge(id) {
    return this.edges.get(id);
  }

  getAllEdges() {
    return Array.from(this.edges.values());
  }

  getOutgoingEdges(nodeId) {
    const edgeIds = this.adjacency.get(nodeId) || new Set();
    return Array.from(edgeIds).map(id => this.edges.get(id)).filter(Boolean);
  }

  getIncomingEdges(nodeId) {
    const edgeIds = this.reverseAdjacency.get(nodeId) || new Set();
    return Array.from(edgeIds).map(id => this.edges.get(id)).filter(Boolean);
  }

  /**
   * Traverses all downstream nodes starting from startNodeId using BFS.
   * Cycle-safe: tracks visited nodes and edges to support feedback loops.
   * 
   * Returns:
   * - orderedNodes: array of downstream node objects ordered by depth/encounter
   * - affectedEdges: array of traversed edge objects
   * - depthMap: Map of nodeId -> integer depth
   * - cycleDetected: boolean indicating if traversal reached back to an ancestor
   * - summaryByCategory: summary of component categories affected
   */
  getDownstreamImpact(startNodeId) {
    if (!this.nodes.has(startNodeId)) {
      return {
        originNode: null,
        affectedNodes: [],
        affectedEdges: [],
        depthMap: new Map(),
        cycleDetected: false,
        summaryByCategory: {}
      };
    }

    const originNode = this.nodes.get(startNodeId);
    const visitedNodes = new Set([startNodeId]);
    const visitedEdges = new Set();
    const orderedNodes = [];
    const depthMap = new Map();
    depthMap.set(startNodeId, 0);

    let cycleDetected = false;

    // Queue of { nodeId, depth }
    const queue = [{ nodeId: startNodeId, depth: 0 }];

    while (queue.length > 0) {
      const { nodeId, depth } = queue.shift();
      const outgoingEdges = this.getOutgoingEdges(nodeId);

      for (const edge of outgoingEdges) {
        visitedEdges.add(edge.id);
        const targetId = edge.target;

        if (targetId === startNodeId) {
          cycleDetected = true;
        }

        if (!visitedNodes.has(targetId)) {
          visitedNodes.add(targetId);
          const targetNode = this.nodes.get(targetId);
          if (targetNode) {
            orderedNodes.push(targetNode);
            depthMap.set(targetId, depth + 1);
            queue.push({ nodeId: targetId, depth: depth + 1 });
          }
        }
      }
    }

    // Group summary by category and type
    const summaryByCategory = {};
    for (const node of orderedNodes) {
      const cat = node.category || 'other';
      if (!summaryByCategory[cat]) {
        summaryByCategory[cat] = [];
      }
      summaryByCategory[cat].push(node);
    }

    const affectedEdges = Array.from(visitedEdges).map(id => this.edges.get(id)).filter(Boolean);

    return {
      originNode,
      affectedNodes: orderedNodes,
      affectedEdges,
      depthMap,
      cycleDetected,
      summaryByCategory
    };
  }

  /**
   * Detects cycles in the graph (e.g. feedback loops).
   */
  findCycles() {
    const visited = new Set();
    const recStack = new Set();
    const cycleNodes = new Set();

    const dfs = (nodeId) => {
      visited.add(nodeId);
      recStack.add(nodeId);

      const outgoing = this.getOutgoingEdges(nodeId);
      for (const edge of outgoing) {
        const neighbor = edge.target;
        if (!visited.has(neighbor)) {
          if (dfs(neighbor)) return true;
        } else if (recStack.has(neighbor)) {
          cycleNodes.add(nodeId);
          cycleNodes.add(neighbor);
          return true;
        }
      }

      recStack.delete(nodeId);
      return false;
    };

    for (const node of this.nodes.values()) {
      if (!visited.has(node.id)) {
        dfs(node.id);
      }
    }

    return Array.from(cycleNodes);
  }
}
