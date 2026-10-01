/**
 * renderer.js - SVG Architecture Graph Renderer
 * 
 * Renders engineering-diagram style nodes, orthogonal/bezier edges,
 * animated feedback loops, and staggered CACE change-impact propagation.
 * 
 * Design:
 * - Academic systems engineering diagram (not generic SaaS AI)
 * - Restrained palette: #F4F1EA, #FAF9F5, #252525, #6B6861, #C9C5BC, #59636E, #B07A32, #A94A3A
 * - Clear, readable typography and crisp borders
 * - Fixed SVG coordinate safety (no CSS transform collision on SVG <g>)
 */

export class GraphRenderer {
  constructor(svgElement, graph, state) {
    this.svg = svgElement;
    this.graph = graph;
    this.state = state;
    
    // Viewport dimensions (generous coordinate space)
    this.viewWidth = 1560;
    this.viewHeight = 630;

    // Node layout constants (generous width prevents any label overflow)
    this.nodeWidth = 180;
    this.nodeHeight = 64;

    this.initSvg();
  }

  initSvg() {
    this.svg.setAttribute('viewBox', `0 0 ${this.viewWidth} ${this.viewHeight}`);
    this.svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    this.svg.innerHTML = `
      <defs>
        <!-- Standard Arrow Marker -->
        <marker id="marker-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#59636E" />
        </marker>
        <!-- Affected / Cascade Arrow Marker -->
        <marker id="marker-arrow-affected" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#A94A3A" />
        </marker>
        <!-- Warning / Undeclared Marker -->
        <marker id="marker-arrow-warning" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#B07A32" />
        </marker>
        <!-- Healthy Marker -->
        <marker id="marker-arrow-healthy" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#4F6F52" />
        </marker>
      </defs>
      <g id="edges-layer"></g>
      <g id="feedback-layer"></g>
      <g id="nodes-layer"></g>
    `;

    this.edgesLayer = this.svg.querySelector('#edges-layer');
    this.feedbackLayer = this.svg.querySelector('#feedback-layer');
    this.nodesLayer = this.svg.querySelector('#nodes-layer');
  }

  /**
   * Computes clean, un-crowded 2D positions for all active nodes.
   */
  calculateLayout() {
    const nodes = this.graph.getAllNodes();
    const positions = new Map();
    const nodeCount = nodes.length;

    // Check if we are in Stage 0 (only data, model, prediction)
    const isStage0 = nodeCount === 3 &&
      this.graph.getNode('data') &&
      this.graph.getNode('model') &&
      this.graph.getNode('prediction') &&
      this.state.featureCount === 0 &&
      this.state.pipelineStagesCount === 0;

    if (isStage0) {
      // Balanced 3-node clean linear layout
      positions.set('data', { x: 240, y: 280 });
      positions.set('model', { x: 780, y: 280 });
      positions.set('prediction', { x: 1320, y: 280 });
      return positions;
    }

    // Dynamic Multi-Column Grid Layout
    // Adapt column X positions based on active stages
    const hasPipeline = this.state.pipelineStagesCount > 0;
    const hasCorrection = !!this.graph.getNode('correction_model');
    const hasConsumers = this.state.hasUndeclaredConsumer;

    const dataX = hasPipeline ? 110 : 220;
    const pipelineX = 320;
    const featuresX = hasPipeline ? 540 : (this.state.featureCount > 0 ? 500 : 450);
    const modelX = 770;
    const correctionX = 990;
    const predictionX = hasCorrection ? 1210 : (hasConsumers ? 1100 : 1080);
    const consumersX = 1430;

    // 1. Data Node
    positions.set('data', { x: dataX, y: 270 });

    // 2. Pipeline Stages (stacked in Col 1)
    const pipelineNodes = ['pipe_etl_1', 'pipe_etl_2', 'pipe_feature_proc'].filter(id => this.graph.getNode(id));
    if (pipelineNodes.length === 1) {
      positions.set(pipelineNodes[0], { x: pipelineX, y: 270 });
    } else if (pipelineNodes.length === 2) {
      positions.set(pipelineNodes[0], { x: pipelineX, y: 190 });
      positions.set(pipelineNodes[1], { x: pipelineX, y: 350 });
    } else if (pipelineNodes.length === 3) {
      positions.set(pipelineNodes[0], { x: pipelineX, y: 130 });
      positions.set(pipelineNodes[1], { x: pipelineX, y: 270 });
      positions.set(pipelineNodes[2], { x: pipelineX, y: 410 });
    }

    // 3. Features (stacked in Col 2)
    const featureNodes = ['feat_a', 'feat_b', 'feat_c', 'feat_d'].filter(id => this.graph.getNode(id));
    if (featureNodes.length === 1) {
      positions.set(featureNodes[0], { x: featuresX, y: 270 });
    } else if (featureNodes.length === 2) {
      positions.set(featureNodes[0], { x: featuresX, y: 200 });
      positions.set(featureNodes[1], { x: featuresX, y: 340 });
    } else if (featureNodes.length === 3) {
      positions.set(featureNodes[0], { x: featuresX, y: 140 });
      positions.set(featureNodes[1], { x: featuresX, y: 270 });
      positions.set(featureNodes[2], { x: featuresX, y: 400 });
    } else if (featureNodes.length === 4) {
      positions.set(featureNodes[0], { x: featuresX, y: 100 });
      positions.set(featureNodes[1], { x: featuresX, y: 210 });
      positions.set(featureNodes[2], { x: featuresX, y: 320 });
      positions.set(featureNodes[3], { x: featuresX, y: 430 });
    }

    // 4. ML Model
    positions.set('model', { x: modelX, y: 270 });

    // 5. Correction Model (if present)
    if (hasCorrection) {
      positions.set('correction_model', { x: correctionX, y: 270 });
    }

    // 6. Prediction
    positions.set('prediction', { x: predictionX, y: 270 });

    // Active Users (below prediction for feedback loop flow)
    if (this.graph.getNode('users')) {
      positions.set('users', { x: predictionX, y: 430 });
    }

    // 7. Consumers (stacked in Col 6)
    const consumerNodes = ['consumer_dashboard', 'consumer_ranking', 'consumer_unknown'].filter(id => this.graph.getNode(id));
    if (consumerNodes.length > 0) {
      positions.set('consumer_dashboard', { x: consumersX, y: 130 });
      positions.set('consumer_ranking', { x: consumersX, y: 270 });
      positions.set('consumer_unknown', { x: consumersX, y: 410 });
    }

    return positions;
  }

  /**
   * Main render function called when state or highlights change.
   */
  render() {
    const positions = this.calculateLayout();
    const activeHighlight = this.state.activeHighlight;

    // Clear previous elements
    this.edgesLayer.innerHTML = '';
    this.feedbackLayer.innerHTML = '';
    this.nodesLayer.innerHTML = '';

    // Render Edges
    const edges = this.graph.getAllEdges();
    for (const edge of edges) {
      const srcPos = positions.get(edge.source);
      const tgtPos = positions.get(edge.target);

      if (!srcPos || !tgtPos) continue;

      if (edge.routing === 'loop' || edge.type === 'feedback_loop') {
        this.renderFeedbackEdge(edge, srcPos, tgtPos, activeHighlight);
      } else {
        this.renderStandardEdge(edge, srcPos, tgtPos, activeHighlight);
      }
    }

    // Render Nodes
    const nodes = this.graph.getAllNodes();
    for (const node of nodes) {
      const pos = positions.get(node.id);
      if (!pos) continue;
      this.renderNode(node, pos, activeHighlight);
    }
  }

  renderStandardEdge(edge, srcPos, tgtPos, activeHighlight) {
    const isTraversed = activeHighlight?.affectedEdges?.some(e => e.id === edge.id);
    const isOriginEdge = activeHighlight?.originNode && edge.source === activeHighlight.originNode.id;

    // Source exit point (right edge of source box)
    const x1 = srcPos.x + this.nodeWidth / 2;
    const y1 = srcPos.y;

    // Target enter point (left edge of target box)
    const x2 = tgtPos.x - this.nodeWidth / 2;
    const y2 = tgtPos.y;

    // If target is directly below (e.g. prediction -> users)
    const isVertical = Math.abs(srcPos.x - tgtPos.x) < 30 && tgtPos.y > srcPos.y;
    let d = '';

    if (isVertical) {
      const vx1 = srcPos.x;
      const vy1 = srcPos.y + this.nodeHeight / 2;
      const vx2 = tgtPos.x;
      const vy2 = tgtPos.y - this.nodeHeight / 2;
      d = `M ${vx1} ${vy1} L ${vx2} ${vy2}`;
    } else {
      // Orthogonal cubic curve with smooth horizontal tangents
      const dx = Math.abs(x2 - x1) * 0.45;
      d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
    }

    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'edge-group');
    g.setAttribute('data-edge-id', edge.id);

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', d);
    path.setAttribute('fill', 'none');

    // Status styling
    let strokeColor = '#59636E';
    let markerId = 'marker-arrow';
    let strokeWidth = '1.75';

    if (isTraversed || isOriginEdge) {
      strokeColor = '#A94A3A';
      markerId = 'marker-arrow-affected';
      strokeWidth = '2.5';
      g.classList.add('edge-affected');
    } else if (edge.status === 'warning' || edge.type === 'correction') {
      strokeColor = '#B07A32';
      markerId = 'marker-arrow-warning';
    }

    path.setAttribute('stroke', strokeColor);
    path.setAttribute('stroke-width', strokeWidth);
    path.setAttribute('marker-end', `url(#${markerId})`);

    if (edge.dashed) {
      path.setAttribute('stroke-dasharray', '4, 4');
    }

    g.appendChild(path);

    // Render edge badge only if there is sufficient horizontal clearance (>85px)
    if (edge.label && Math.abs(x2 - x1) >= 85 && (edge.status === 'warning' || edge.dashed)) {
      const midX = (x1 + x2) / 2;
      const midY = (y1 + y2) / 2 - 8;
      
      const badgeG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      badgeG.setAttribute('transform', `translate(${midX}, ${midY})`);

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', '0');
      text.setAttribute('y', '0');
      text.setAttribute('class', 'edge-label');
      text.setAttribute('text-anchor', 'middle');
      text.textContent = edge.label;

      badgeG.appendChild(text);
      g.appendChild(badgeG);
    }

    this.edgesLayer.appendChild(g);
  }

  /**
   * Renders the feedback loop spanning from Users/Prediction back to Data across bottom.
   */
  renderFeedbackEdge(edge, srcPos, tgtPos, activeHighlight) {
    const isTraversed = activeHighlight?.affectedEdges?.some(e => e.id === edge.id);

    // Exit bottom of source (Users)
    const x1 = srcPos.x;
    const y1 = srcPos.y + this.nodeHeight / 2;

    // Enter bottom of target (Data)
    const x2 = tgtPos.x;
    const y2 = tgtPos.y + this.nodeHeight / 2;

    const bottomY = 560; // Clean lower conduit line

    const radius = 14;
    const d = `
      M ${x1} ${y1}
      V ${bottomY - radius}
      Q ${x1} ${bottomY} ${x1 - radius} ${bottomY}
      H ${x2 + radius}
      Q ${x2} ${bottomY} ${x2} ${bottomY - radius}
      V ${y2 + 8}
    `;

    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'feedback-loop-group');

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', d);
    path.setAttribute('fill', 'none');
    
    const strokeColor = isTraversed ? '#A94A3A' : '#B07A32';
    const marker = isTraversed ? 'marker-arrow-affected' : 'marker-arrow-warning';

    path.setAttribute('stroke', strokeColor);
    path.setAttribute('stroke-width', isTraversed ? '2.5' : '2');
    path.setAttribute('stroke-dasharray', '6, 5');
    path.setAttribute('class', 'feedback-animated-path');
    path.setAttribute('marker-end', `url(#${marker})`);

    g.appendChild(path);

    // Feedback Tag / Annotation in the middle of the bottom conduit
    const tagX = (x1 + x2) / 2;
    const tagY = bottomY;

    const tagG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    tagG.setAttribute('transform', `translate(${tagX}, ${tagY})`);

    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', '-130');
    rect.setAttribute('y', '-12');
    rect.setAttribute('width', '260');
    rect.setAttribute('height', '24');
    rect.setAttribute('rx', '3');
    rect.setAttribute('fill', '#FAF9F5');
    rect.setAttribute('stroke', strokeColor);
    rect.setAttribute('stroke-width', '1.2');

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', '0');
    text.setAttribute('y', '4');
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('class', 'feedback-tag-text');
    text.setAttribute('fill', strokeColor);
    text.textContent = '↺ FUTURE TRAINING DATA (FEEDBACK LOOP)';

    tagG.appendChild(rect);
    tagG.appendChild(text);
    g.appendChild(tagG);

    this.feedbackLayer.appendChild(g);
  }

  /**
   * Renders an individual engineering node box.
   */
  renderNode(node, pos, activeHighlight) {
    const isOrigin = activeHighlight?.originNode?.id === node.id;
    const isAffected = activeHighlight?.affectedNodes?.some(n => n.id === node.id);
    const depth = activeHighlight?.depthMap?.get(node.id) ?? 0;

    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', `node-group node-type-${node.type}`);
    g.setAttribute('data-node-id', node.id);
    // Explicit static translation - NEVER animated by CSS transform
    g.setAttribute('transform', `translate(${pos.x - this.nodeWidth / 2}, ${pos.y - this.nodeHeight / 2})`);

    // State classes
    if (isOrigin) g.classList.add('node-origin');
    if (isAffected) g.classList.add('node-affected');
    if (node.status === 'warning' || node.meta?.undeclared) g.classList.add('node-warning');

    // Box background
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', this.nodeWidth);
    rect.setAttribute('height', this.nodeHeight);
    rect.setAttribute('rx', '4');
    rect.setAttribute('class', 'node-box');

    // Color & stroke logic according to design system
    let strokeColor = '#59636E';
    let fillColor = '#FFFFFF';
    let strokeWidth = '1.5';

    if (isOrigin) {
      strokeColor = '#A94A3A';
      fillColor = '#FDF2F0';
      strokeWidth = '2.5';
    } else if (isAffected) {
      strokeColor = '#A94A3A';
      fillColor = '#FDF2F0';
      strokeWidth = '2';
      rect.style.animationDelay = `${depth * 140}ms`;
    } else if (node.status === 'warning' || node.meta?.undeclared) {
      strokeColor = '#B07A32';
      fillColor = '#FFFDF7';
      strokeWidth = '2';
    } else if (node.type === 'model') {
      strokeColor = '#252525';
      fillColor = '#FAF9F5';
      strokeWidth = '2';
    } else if (node.type === 'correction_model') {
      strokeColor = '#B07A32';
      fillColor = '#FFFDF7';
      strokeWidth = '2';
    }

    rect.setAttribute('stroke', strokeColor);
    rect.setAttribute('fill', fillColor);
    rect.setAttribute('stroke-width', strokeWidth);

    g.appendChild(rect);

    // Top Category Header Stripe
    const headerLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    headerLine.setAttribute('x1', '0');
    headerLine.setAttribute('y1', '20');
    headerLine.setAttribute('x2', this.nodeWidth);
    headerLine.setAttribute('y2', '20');
    headerLine.setAttribute('stroke', strokeColor);
    headerLine.setAttribute('stroke-width', '1');
    headerLine.setAttribute('opacity', '0.35');
    g.appendChild(headerLine);

    // Left Category / Type Label
    const typeLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    typeLabel.setAttribute('x', '8');
    typeLabel.setAttribute('y', '14.5');
    typeLabel.setAttribute('class', 'node-type-label');
    typeLabel.setAttribute('fill', strokeColor);
    typeLabel.textContent = this.formatTypeLabel(node);
    g.appendChild(typeLabel);

    // Right Pill Badge for Status / Hops (Isolated pill prevents ANY text overlap)
    if (isAffected) {
      const pillWidth = 48;
      const pillX = this.nodeWidth - pillWidth - 6;

      const pillRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      pillRect.setAttribute('x', pillX);
      pillRect.setAttribute('y', '4');
      pillRect.setAttribute('width', pillWidth);
      pillRect.setAttribute('height', '13');
      pillRect.setAttribute('rx', '2');
      pillRect.setAttribute('fill', '#FDF2F0');
      pillRect.setAttribute('stroke', '#A94A3A');
      pillRect.setAttribute('stroke-width', '0.8');

      const affTag = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      affTag.setAttribute('x', pillX + pillWidth / 2);
      affTag.setAttribute('y', '13.5');
      affTag.setAttribute('text-anchor', 'middle');
      affTag.setAttribute('class', 'node-status-tag');
      affTag.setAttribute('fill', '#A94A3A');
      affTag.textContent = `+${depth} HOP`;

      g.appendChild(pillRect);
      g.appendChild(affTag);
    } else if (node.meta?.undeclared) {
      const pillWidth = 74;
      const pillX = this.nodeWidth - pillWidth - 6;

      const pillRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      pillRect.setAttribute('x', pillX);
      pillRect.setAttribute('y', '4');
      pillRect.setAttribute('width', pillWidth);
      pillRect.setAttribute('height', '13');
      pillRect.setAttribute('rx', '2');
      pillRect.setAttribute('fill', '#FFFDF7');
      pillRect.setAttribute('stroke', '#B07A32');
      pillRect.setAttribute('stroke-width', '0.8');

      const unkTag = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      unkTag.setAttribute('x', pillX + pillWidth / 2);
      unkTag.setAttribute('y', '13.5');
      unkTag.setAttribute('text-anchor', 'middle');
      unkTag.setAttribute('class', 'node-status-tag');
      unkTag.setAttribute('fill', '#B07A32');
      unkTag.textContent = '⚠ UNMONITORED';

      g.appendChild(pillRect);
      g.appendChild(unkTag);
    }

    // Node Title (Primary Label) - perfectly proportioned
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    title.setAttribute('x', '10');
    title.setAttribute('y', '38');
    title.setAttribute('class', 'node-title');
    title.setAttribute('fill', isOrigin || isAffected ? '#A94A3A' : '#252525');
    title.textContent = node.label;
    g.appendChild(title);

    // Node Sublabel
    if (node.sublabel) {
      const sub = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      sub.setAttribute('x', '10');
      sub.setAttribute('y', '53');
      sub.setAttribute('class', 'node-sublabel');
      sub.setAttribute('fill', '#6B6861');
      sub.textContent = node.sublabel;
      g.appendChild(sub);
    }

    // Interactive tooltip / click handling
    g.style.cursor = 'pointer';
    g.addEventListener('click', () => {
      if (node.type === 'feature' || node.id === 'feat_a') {
        this.state.triggerChangeFeatureA();
      }
    });

    this.nodesLayer.appendChild(g);
  }

  formatTypeLabel(node) {
    switch (node.type) {
      case 'dataset': return 'RAW DATASET';
      case 'etl': return 'PIPELINE';
      case 'feature': return 'FEATURE';
      case 'model': return 'PRIMARY MODEL';
      case 'correction_model': return 'PATCH MODEL';
      case 'prediction': return 'SERVING API';
      case 'consumer': return node.meta?.undeclared ? 'EXTERNAL' : 'CONSUMER';
      case 'user': return 'ACTIVE AGENT';
      default: return 'COMPONENT';
    }
  }
}
