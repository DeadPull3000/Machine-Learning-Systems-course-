/**
 * debt-analysis.js - Static Analysis Simulation Engine
 * 
 * Inspects the current dependency graph for ML Technical Debt anti-patterns
 * identified in Sculley et al. (2015).
 * 
 * IMPORTANT: This is an illustrative static-analysis report (a proposed extension
 * inspired by the paper), not something that the original paper explicitly implements.
 */

export class DebtAnalyzer {
  constructor(graph, state) {
    this.graph = graph;
    this.state = state;
  }

  analyze() {
    const findings = [];
    const nodes = this.graph.getAllNodes();
    const edges = this.graph.getAllEdges();

    // 1. CACE Risk (Changing Anything Changes Everything)
    const featA = this.graph.getNode('feat_a');
    if (featA) {
      const impact = this.graph.getDownstreamImpact('feat_a');
      const affectedCount = impact.affectedNodes.length;
      if (affectedCount >= 3) {
        findings.push({
          severity: 'HIGH',
          title: 'CACE RISK (ENTANGLEMENT)',
          paperRef: 'Section 2: Complex Models Lose Boundaries',
          description: `Feature A directly or indirectly affects ${affectedCount} downstream components.`,
          detail: 'No inputs are truly independent. Modifying, tuning, or pruning this feature cascades across all subsequent transformations and serving layers.',
          targetNodes: ['feat_a', ...impact.affectedNodes.map(n => n.id)]
        });
      }
    }

    // 2. Correction Cascade
    const correctionNode = this.graph.getNode('correction_model');
    if (correctionNode) {
      findings.push({
        severity: 'HIGH',
        title: 'CORRECTION CASCADE',
        paperRef: 'Section 2: Correction Cascades',
        description: "Model A → Correction Model A' pipeline detected.",
        detail: "Patching edge cases with a downstream secondary model masks root distribution shifts and traps the system in a cascading learning dilemma when retraining Model A.",
        targetNodes: ['model', 'correction_model', 'prediction']
      });
    }

    // 3. Undeclared Consumers
    const undeclaredNodes = nodes.filter(n => n.meta?.undeclared || n.id === 'consumer_unknown');
    if (undeclaredNodes.length > 0) {
      findings.push({
        severity: 'HIGH',
        title: 'UNDECLARED CONSUMER',
        paperRef: 'Section 3: Undeclared Consumers',
        description: 'Prediction stream is consumed by an unmonitored external service.',
        detail: 'Without explicit contracts or access control, updating the model or schema risks breaking silent downstream systems without notification or recourse.',
        targetNodes: undeclaredNodes.map(n => n.id)
      });
    }

    // 4. Feedback Loops (Direct & Hidden)
    const feedbackEdges = edges.filter(e => e.type === 'feedback_loop' || e.routing === 'loop');
    const cycles = this.graph.findCycles();
    if (feedbackEdges.length > 0 || cycles.length > 0) {
      findings.push({
        severity: 'MEDIUM',
        title: 'DIRECT FEEDBACK LOOP',
        paperRef: 'Section 4: Data Dependencies (Feedback Loops)',
        description: 'Model predictions directly influence future training datasets.',
        detail: 'Serving decisions alter end-user behaviour, creating self-fulfilling bias in subsequent training sets and rendering traditional offline validation misleading.',
        targetNodes: ['users', 'data', 'prediction']
      });
    }

    // 5. Pipeline Jungles & Glue Code
    const pipelineNodes = nodes.filter(n => n.type === 'etl');
    if (pipelineNodes.length >= 2) {
      findings.push({
        severity: 'MEDIUM',
        title: 'PIPELINE COMPLEXITY / GLUE CODE',
        paperRef: 'Section 5: ML-System Anti-Patterns (Pipeline Jungles)',
        description: `${pipelineNodes.length} sequential data transformation stages detected.`,
        detail: 'Handcrafted ETL pipelines and custom feature scrapers create integration friction, config drift, and testing debt around an otherwise simple model.',
        targetNodes: pipelineNodes.map(n => n.id)
      });
    }

    // 6. Unstable Data Dependencies
    if (this.state.featureCount >= 3) {
      findings.push({
        severity: 'MEDIUM',
        title: 'UNSTABLE FEATURE DEPENDENCY',
        paperRef: 'Section 4: Unstable Data Dependencies',
        description: 'Multiple input signals derived from external operational streams.',
        detail: 'Features can quietly shift their semantic meaning, calibration, or distribution upstream without throwing syntactic errors.',
        targetNodes: ['feat_c', 'feat_d']
      });
    }

    // Sort findings: HIGH first, then MEDIUM, then LOW
    const severityWeight = { 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1 };
    findings.sort((a, b) => severityWeight[b.severity] - severityWeight[a.severity]);

    return {
      timestamp: new Date().toLocaleTimeString(),
      totalIssues: findings.length,
      highCount: findings.filter(f => f.severity === 'HIGH').length,
      mediumCount: findings.filter(f => f.severity === 'MEDIUM').length,
      findings
    };
  }
}
