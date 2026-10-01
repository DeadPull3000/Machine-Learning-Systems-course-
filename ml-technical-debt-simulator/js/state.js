/**
 * state.js - Simulation State Machine and Scenario Manager
 * 
 * Manages presentation stages, simulated metrics (accuracy, debt events),
 * step-by-step narration, and graph modifications.
 * 
 * NOTE: As emphasized by Sculley et al., technical debt has no strict metric.
 * Accuracy values and debt counters are purely illustrative presentation tools.
 */

export class SimulationState {
  constructor(graph) {
    this.graph = graph;
    this.listeners = new Set();
    
    // Core metrics
    this.accuracy = 92.4;
    this.baseAccuracy = 92.4;
    this.debtEvents = 0;
    
    // Track added entities
    this.featureCount = 0;
    this.hasCorrectionModel = false;
    this.hasUndeclaredConsumer = false;
    this.hasFeedbackLoop = false;
    this.pipelineStagesCount = 0; // 0 to 3
    
    // Presentation narrative state
    this.currentNarrative = {
      title: "Clean System",
      concept: "Minimal Baseline",
      quote: "One model, one input pipeline, one prediction interface.",
      detail: "Acc: 92.4% | Low Coupling | Clear Boundaries",
      badge: "Stage 0"
    };

    this.isPresentationMode = false;
    this.activeHighlight = null; // currently highlighted nodes/edges from change impact
    this.isAnalyzingDebt = false;

    // Initialize with Stage 0
    this.initStage0();
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify(event, payload = {}) {
    this.listeners.forEach(fn => fn(event, payload, this));
  }

  /**
   * Resets graph and state to Stage 0 (Clean System).
   */
  reset() {
    this.graph.clear();
    this.featureCount = 0;
    this.hasCorrectionModel = false;
    this.hasUndeclaredConsumer = false;
    this.hasFeedbackLoop = false;
    this.pipelineStagesCount = 0;
    this.debtEvents = 0;
    this.accuracy = this.baseAccuracy;
    this.activeHighlight = null;
    this.isAnalyzingDebt = false;

    this.initStage0();
    this.notify('state_changed', { action: 'reset' });
  }

  /**
   * Stage 0: Clean System
   * DATA -> ML MODEL -> PREDICTION
   */
  initStage0() {
    this.graph.clear();

    this.graph.addNode({
      id: 'data',
      label: 'DATA',
      sublabel: 'Customer Events v1.0',
      type: 'dataset',
      category: 'input',
      col: 0,
      row: 0
    });

    this.graph.addNode({
      id: 'model',
      label: 'ML MODEL',
      sublabel: 'Logistic Regression v1',
      type: 'model',
      category: 'model',
      col: 3,
      row: 0
    });

    this.graph.addNode({
      id: 'prediction',
      label: 'PREDICTION',
      sublabel: 'Risk Score API / JSON',
      type: 'prediction',
      category: 'serving',
      col: 5,
      row: 0
    });

    this.graph.addEdge({
      id: 'data->model',
      source: 'data',
      target: 'model',
      label: 'clean tabular stream',
      type: 'data_dependency'
    });

    this.graph.addEdge({
      id: 'model->prediction',
      source: 'model',
      target: 'prediction',
      label: 'probabilities',
      type: 'data_dependency'
    });

    this.currentNarrative = {
      title: "Clean System",
      concept: "Minimal Baseline",
      quote: "One model, one input pipeline, one prediction interface.",
      detail: "Low coupling, clear system boundaries, high visibility.",
      badge: "Stage 0"
    };

    this.accuracy = 92.4;
    this.debtEvents = 0;
  }

  /**
   * Stage 1: Add a Feature (Feature A, B, C, D)
   * Represents CACE (Changing Anything Changes Everything)
   */
  addFeature() {
    if (this.featureCount >= 4) return false;

    const featureConfigs = [
      { id: 'feat_a', label: 'FEATURE A', sublabel: 'Account Age & History' },
      { id: 'feat_b', label: 'FEATURE B', sublabel: '30-Day Activity Freq' },
      { id: 'feat_c', label: 'FEATURE C', sublabel: 'Geographic Risk Code' },
      { id: 'feat_d', label: 'FEATURE D', sublabel: 'Device Fingerprint' }
    ];

    const feat = featureConfigs[this.featureCount];
    this.featureCount++;

    // If this is the first feature, remove direct data->model edge if present,
    // or rewire data to features
    if (this.featureCount === 1) {
      if (this.graph.getEdge('data->model')) {
        this.graph.removeEdge('data->model');
      }
    }

    // Determine input source (either data or latest pipeline stage)
    const sourceNodeId = this.getInputSourceId();

    this.graph.addNode({
      id: feat.id,
      label: feat.label,
      sublabel: feat.sublabel,
      type: 'feature',
      category: 'processing',
      col: 2,
      row: this.featureCount - 1
    });

    this.graph.addEdge({
      id: `${sourceNodeId}->${feat.id}`,
      source: sourceNodeId,
      target: feat.id,
      type: 'data_dependency'
    });

    this.graph.addEdge({
      id: `${feat.id}->model`,
      source: feat.id,
      target: 'model',
      type: 'data_dependency'
    });

    // Update metrics
    const accuracySteps = [93.0, 93.6, 94.1, 94.5];
    this.accuracy = accuracySteps[this.featureCount - 1];
    this.debtEvents += 1;

    this.currentNarrative = {
      title: "Entanglement & CACE",
      concept: "Changing Anything Changes Everything",
      quote: "Accuracy improved, but the model became more tightly coupled.",
      detail: `Feature ${feat.label.replace('FEATURE ', '')} added. Feature distributions and weights are now co-dependent.`,
      badge: `Stage 1 (+Feature ${feat.label.replace('FEATURE ', '')})`
    };

    this.notify('state_changed', { action: 'add_feature', featureId: feat.id });
    return true;
  }

  /**
   * Helper: gets current node that feeds into features
   */
  getInputSourceId() {
    if (this.pipelineStagesCount === 3) return 'pipe_feature_proc';
    if (this.pipelineStagesCount === 2) return 'pipe_etl_2';
    if (this.pipelineStagesCount === 1) return 'pipe_etl_1';
    return 'data';
  }

  /**
   * Stage 2: Add Correction Model (Model A')
   * Represents Correction Cascades
   */
  addCorrectionModel() {
    if (this.hasCorrectionModel) return false;
    this.hasCorrectionModel = true;

    // Reroute: ML Model -> Correction Model -> Prediction
    // Remove model->prediction edge
    this.graph.removeEdge('model->prediction');

    this.graph.addNode({
      id: 'correction_model',
      label: 'CORRECTION MODEL',
      sublabel: "Model A' (Edge Patch)",
      type: 'correction_model',
      category: 'model',
      col: 4,
      row: 0
    });

    this.graph.addEdge({
      id: 'model->correction_model',
      source: 'model',
      target: 'correction_model',
      label: 'raw inference',
      type: 'correction'
    });

    this.graph.addEdge({
      id: 'correction_model->prediction',
      source: 'correction_model',
      target: 'prediction',
      label: 'calibrated score',
      type: 'data_dependency'
    });

    this.accuracy = +(this.accuracy + 0.3).toFixed(1);
    this.debtEvents += 2;

    this.currentNarrative = {
      title: "Correction Cascade",
      concept: "Patching Mistakes Post-Hoc",
      quote: "A quick correction creates another model dependency.",
      detail: "Instead of retraining model v1, a secondary model patches corner cases, creating a cascading learning dilemma.",
      badge: "Stage 2"
    };

    this.notify('state_changed', { action: 'add_correction_model' });
    return true;
  }

  /**
   * Stage 3: Add Consumers (Dashboard, Ranking Service, UNKNOWN / UNDECLARED)
   * Represents Undeclared Consumers
   */
  addUndeclaredConsumer() {
    if (this.hasUndeclaredConsumer) return false;
    this.hasUndeclaredConsumer = true;

    // Add 3 consumers connected to prediction
    this.graph.addNode({
      id: 'consumer_dashboard',
      label: 'DASHBOARD',
      sublabel: 'Executive Analytics View',
      type: 'consumer',
      category: 'consumer',
      col: 6,
      row: 0
    });

    this.graph.addNode({
      id: 'consumer_ranking',
      label: 'RANKING SERVICE',
      sublabel: 'Feed Re-ranking Engine',
      type: 'consumer',
      category: 'consumer',
      col: 6,
      row: 1
    });

    this.graph.addNode({
      id: 'consumer_unknown',
      label: 'UNKNOWN / UNDECLARED',
      sublabel: 'Third-Party Consumer',
      type: 'consumer',
      category: 'consumer',
      status: 'warning',
      meta: { undeclared: true },
      col: 6,
      row: 2
    });

    this.graph.addEdge({
      id: 'prediction->consumer_dashboard',
      source: 'prediction',
      target: 'consumer_dashboard',
      label: 'batch read',
      type: 'consumer_feed'
    });

    this.graph.addEdge({
      id: 'prediction->consumer_ranking',
      source: 'prediction',
      target: 'consumer_ranking',
      label: 'real-time RPC',
      type: 'consumer_feed'
    });

    this.graph.addEdge({
      id: 'prediction->consumer_unknown',
      source: 'prediction',
      target: 'consumer_unknown',
      label: 'unmonitored pull',
      type: 'consumer_feed',
      status: 'warning',
      dashed: true
    });

    this.debtEvents += 2;

    this.currentNarrative = {
      title: "Undeclared Consumers",
      concept: "Hidden Coupling & Silent Dependencies",
      quote: "Who else is using this prediction? Changing the model is now high risk.",
      detail: "Without strict access contracts or SLAs, changing model output distributions can silently break unexpected downstream clients.",
      badge: "Stage 3"
    };

    this.notify('state_changed', { action: 'add_undeclared_consumer' });
    return true;
  }

  /**
   * Stage 4: Add Feedback Loop
   * Represents Direct / Hidden Feedback Loops
   */
  addFeedbackLoop() {
    if (this.hasFeedbackLoop) return false;
    this.hasFeedbackLoop = true;

    // Add USERS node if not present
    if (!this.graph.getNode('users')) {
      this.graph.addNode({
        id: 'users',
        label: 'ACTIVE USERS',
        sublabel: 'End-user clicks & conversions',
        type: 'user',
        category: 'consumer',
        col: 5,
        row: 1
      });

      this.graph.addEdge({
        id: 'prediction->users',
        source: 'prediction',
        target: 'users',
        label: 'served actions',
        type: 'data_dependency'
      });
    }

    // Connect Users -> Training Data via feedback edge
    this.graph.addEdge({
      id: 'users->data',
      source: 'users',
      target: 'data',
      label: 'FUTURE TRAINING DATA',
      type: 'feedback_loop',
      animated: true,
      dashed: true,
      routing: 'loop',
      status: 'warning'
    });

    this.debtEvents += 2;

    this.currentNarrative = {
      title: "Direct Feedback Loop",
      concept: "Model Influencing Future Ground Truth",
      quote: "The model now influences the data it will learn from.",
      detail: "Predictions shape user actions, which become tomorrow's training labels, creating self-fulfilling drift and hidden biases.",
      badge: "Stage 4"
    };

    this.notify('state_changed', { action: 'add_feedback_loop' });
    return true;
  }

  /**
   * Stage 5: Add Pipeline Stage (up to 3 stages)
   * Represents Pipeline Jungles and Glue Code
   */
  addPipelineStage() {
    if (this.pipelineStagesCount >= 3) return false;
    this.pipelineStagesCount++;

    const stages = [
      {
        id: 'pipe_etl_1',
        label: 'ETL STAGE 1',
        sublabel: 'Log Ingest & Join',
        type: 'etl',
        col: 1,
        row: 0
      },
      {
        id: 'pipe_etl_2',
        label: 'ETL STAGE 2',
        sublabel: 'Aggregate Windows',
        type: 'etl',
        col: 1,
        row: 1
      },
      {
        id: 'pipe_feature_proc',
        label: 'FEATURE PROCESSING',
        sublabel: 'Impute & Normalize',
        type: 'etl',
        col: 1,
        row: 2
      }
    ];

    const currentStage = stages[this.pipelineStagesCount - 1];
    this.graph.addNode(currentStage);

    // Reconfigure edges connecting data -> ETLs -> features/model
    this.reconfigurePipelineEdges();

    this.debtEvents += 1;

    this.currentNarrative = {
      title: "Pipeline Jungles & Glue Code",
      concept: "Infrastructure Accumulation",
      quote: "The model is unchanged. The infrastructure around it is growing.",
      detail: "Massive glue code and custom ETL chains obscure data provenance and introduce integration brittleness.",
      badge: `Stage 5 (Pipeline ${this.pipelineStagesCount}/3)`
    };

    this.notify('state_changed', { action: 'add_pipeline_stage', stageId: currentStage.id });
    return true;
  }

  /**
   * Rewires pipeline connectors cleanly
   */
  reconfigurePipelineEdges() {
    // Collect all active pipeline stage IDs
    const stages = ['pipe_etl_1', 'pipe_etl_2', 'pipe_feature_proc'].slice(0, this.pipelineStagesCount);

    // Remove any direct data outgoing edges to features or model
    const outgoingFromData = this.graph.getOutgoingEdges('data');
    for (const e of outgoingFromData) {
      if (e.target !== 'pipe_etl_1') {
        this.graph.removeEdge(e.id);
      }
    }

    // Connect Data -> ETL 1
    if (stages.length >= 1 && !this.graph.getEdge('data->pipe_etl_1')) {
      this.graph.addEdge({
        id: 'data->pipe_etl_1',
        source: 'data',
        target: 'pipe_etl_1',
        type: 'data_dependency'
      });
    }

    // Connect ETL 1 -> ETL 2
    if (stages.length >= 2 && !this.graph.getEdge('pipe_etl_1->pipe_etl_2')) {
      this.graph.addEdge({
        id: 'pipe_etl_1->pipe_etl_2',
        source: 'pipe_etl_1',
        target: 'pipe_etl_2',
        type: 'data_dependency'
      });
    }

    // Connect ETL 2 -> Feature Processing
    if (stages.length >= 3 && !this.graph.getEdge('pipe_etl_2->pipe_feature_proc')) {
      this.graph.addEdge({
        id: 'pipe_etl_2->pipe_feature_proc',
        source: 'pipe_etl_2',
        target: 'pipe_feature_proc',
        type: 'data_dependency'
      });
    }

    // The tail of the pipeline feeds into features (or model if no features yet)
    const pipelineTail = stages[stages.length - 1];

    // Remove old pipeline tail connections if we upgraded
    if (stages.length === 2 && this.graph.getEdge('pipe_etl_1->model')) {
      this.graph.removeEdge('pipe_etl_1->model');
    }
    if (stages.length === 3 && this.graph.getEdge('pipe_etl_2->model')) {
      this.graph.removeEdge('pipe_etl_2->model');
    }

    // Connect pipeline tail to features
    if (this.featureCount > 0) {
      for (let i = 0; i < this.featureCount; i++) {
        const featId = ['feat_a', 'feat_b', 'feat_c', 'feat_d'][i];
        
        // Remove old incoming edge to feature
        const incoming = this.graph.getIncomingEdges(featId);
        incoming.forEach(edge => this.graph.removeEdge(edge.id));

        // Add new edge from pipelineTail
        this.graph.addEdge({
          id: `${pipelineTail}->${featId}`,
          source: pipelineTail,
          target: featId,
          type: 'data_dependency'
        });
      }
    } else {
      // Connect pipelineTail directly to model
      if (!this.graph.getEdge(`${pipelineTail}->model`)) {
        this.graph.addEdge({
          id: `${pipelineTail}->model`,
          source: pipelineTail,
          target: 'model',
          type: 'data_dependency'
        });
      }
    }
  }

  /**
   * Triggers downstream impact traversal from Feature A (or specified node).
   */
  triggerChangeFeatureA() {
    // If feature A doesn't exist yet, add it automatically
    if (!this.graph.getNode('feat_a')) {
      this.addFeature();
    }

    const impact = this.graph.getDownstreamImpact('feat_a');
    this.activeHighlight = impact;

    this.currentNarrative = {
      title: "CACE Impact Propagation",
      concept: "Dependency Closure of Feature A",
      quote: "Feature A changed. Observe the downstream ripple effect across the entire system.",
      detail: `${impact.affectedNodes.length} downstream components affected across multiple operational boundaries.`,
      badge: "Impact Demo"
    };

    this.notify('change_impact_triggered', { impact });
    return impact;
  }

  clearHighlight() {
    this.activeHighlight = null;
    this.notify('highlight_cleared', {});
  }

  getDebtLevel() {
    if (this.debtEvents <= 1) return { label: 'LOW', color: '#4F6F52', percent: 15 };
    if (this.debtEvents <= 4) return { label: 'MODERATE', color: '#B07A32', percent: 45 };
    if (this.debtEvents <= 7) return { label: 'HIGH', color: '#B07A32', percent: 75 };
    return { label: 'CRITICAL', color: '#A94A3A', percent: 95 };
  }

  isMaxComplexity() {
    return this.graph.getAllNodes().length >= 10 && this.debtEvents >= 6;
  }
}
