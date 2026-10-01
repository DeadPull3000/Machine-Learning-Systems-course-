/**
 * ui.js - Presentation Controller and User Interface Manager
 * 
 * Handles controls, presentation mode, change-impact drawer,
 * technical debt meter, debt analysis report modal, keyboard shortcuts,
 * and academic narrative displays.
 */

import { DebtAnalyzer } from './debt-analysis.js';

export class UIManager {
  constructor(state, renderer) {
    this.state = state;
    this.renderer = renderer;
    this.analyzer = new DebtAnalyzer(this.state.graph, this.state);

    this.cacheDom();
    this.bindEvents();
    this.updateAll();
  }

  cacheDom() {
    // Header & Stats
    this.statAccuracy = document.getElementById('stat-accuracy');
    this.statDebt = document.getElementById('stat-debt');
    this.statNodes = document.getElementById('stat-nodes');
    this.debtMeterFill = document.getElementById('debt-meter-fill');
    this.debtMeterLabel = document.getElementById('debt-meter-label');

    // Controls
    this.btnAddFeature = document.getElementById('btn-add-feature');
    this.btnAddCorrection = document.getElementById('btn-add-correction');
    this.btnAddConsumer = document.getElementById('btn-add-consumer');
    this.btnAddFeedback = document.getElementById('btn-add-feedback');
    this.btnAddPipeline = document.getElementById('btn-add-pipeline');
    this.btnChangeFeatureA = document.getElementById('btn-change-feature-a');
    this.btnRunAnalysis = document.getElementById('btn-run-analysis');
    this.btnReset = document.getElementById('btn-reset');
    this.btnPresentationToggle = document.getElementById('btn-presentation-toggle');
    this.btnNextStep = document.getElementById('btn-next-step');

    // Narrator
    this.narratorBadge = document.getElementById('narrator-badge');
    this.narratorTitle = document.getElementById('narrator-title');
    this.narratorQuote = document.getElementById('narrator-quote');
    this.narratorDetail = document.getElementById('narrator-detail');

    // Impact Panel / Drawer
    this.impactPanel = document.getElementById('impact-panel');
    this.impactCount = document.getElementById('impact-count');
    this.impactList = document.getElementById('impact-list');
    this.btnCloseImpact = document.getElementById('btn-close-impact');

    // Debt Report Modal
    this.reportModal = document.getElementById('report-modal');
    this.reportContent = document.getElementById('report-content');
    this.btnCloseReport = document.getElementById('btn-close-report');

    // Easter Egg
    this.statusEasterEgg = document.getElementById('status-easter-egg');

    // App Container
    this.appContainer = document.getElementById('app-container');
  }

  bindEvents() {
    // Stage Action Buttons
    this.btnAddFeature.addEventListener('click', () => {
      this.state.addFeature();
      this.state.clearHighlight();
    });

    this.btnAddCorrection.addEventListener('click', () => {
      this.state.addCorrectionModel();
      this.state.clearHighlight();
    });

    this.btnAddConsumer.addEventListener('click', () => {
      this.state.addUndeclaredConsumer();
      this.state.clearHighlight();
    });

    this.btnAddFeedback.addEventListener('click', () => {
      this.state.addFeedbackLoop();
      this.state.clearHighlight();
    });

    this.btnAddPipeline.addEventListener('click', () => {
      this.state.addPipelineStage();
      this.state.clearHighlight();
    });

    // Step-by-Step Guided Next Button (Presenter workflow)
    if (this.btnNextStep) {
      this.btnNextStep.addEventListener('click', () => this.advanceNextStep());
    }

    // Change Feature A Demo
    this.btnChangeFeatureA.addEventListener('click', () => {
      const impact = this.state.triggerChangeFeatureA();
      this.showImpactPanel(impact);
    });

    // Run Debt Analysis
    this.btnRunAnalysis.addEventListener('click', () => {
      this.showDebtReport();
    });

    // Reset
    this.btnReset.addEventListener('click', () => {
      this.state.reset();
      this.hideImpactPanel();
      this.hideDebtReport();
    });

    // Presentation Mode Toggle
    this.btnPresentationToggle.addEventListener('click', () => {
      this.togglePresentationMode();
    });

    // Close Impact Panel
    if (this.btnCloseImpact) {
      this.btnCloseImpact.addEventListener('click', () => {
        this.hideImpactPanel();
        this.state.clearHighlight();
        this.renderer.render();
      });
    }

    // Close Report Modal
    if (this.btnCloseReport) {
      this.btnCloseReport.addEventListener('click', () => {
        this.hideDebtReport();
      });
    }

    // Click outside modal to close
    window.addEventListener('click', (e) => {
      if (e.target === this.reportModal) {
        this.hideDebtReport();
      }
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      // Don't trigger if focus is on an input or textarea
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

      if (e.code === 'Space' || e.key === 'ArrowRight') {
        e.preventDefault();
        this.advanceNextStep();
      } else if (e.key === 'p' || e.key === 'P') {
        this.togglePresentationMode();
      } else if (e.key === 'c' || e.key === 'C') {
        const impact = this.state.triggerChangeFeatureA();
        this.showImpactPanel(impact);
      } else if (e.key === 'd' || e.key === 'D') {
        this.showDebtReport();
      } else if (e.key === 'r' || e.key === 'R') {
        this.state.reset();
        this.hideImpactPanel();
        this.hideDebtReport();
      } else if (e.key === 'Escape') {
        this.hideDebtReport();
        this.hideImpactPanel();
        this.state.clearHighlight();
        this.renderer.render();
      }
    });

    // Listen to state changes
    this.state.subscribe((event, payload) => {
      this.updateAll();
      this.renderer.render();
    });
  }

  /**
   * Guides the presenter through the 7 narrative steps seamlessly.
   */
  advanceNextStep() {
    if (this.state.featureCount === 0) {
      this.state.addFeature(); // Feature A
    } else if (this.state.featureCount === 1) {
      this.state.addFeature(); // Feature B
    } else if (this.state.featureCount === 2) {
      this.state.addFeature(); // Feature C
    } else if (this.state.featureCount === 3) {
      this.state.addFeature(); // Feature D
    } else if (!this.state.hasCorrectionModel) {
      this.state.addCorrectionModel();
    } else if (!this.state.hasUndeclaredConsumer) {
      this.state.addUndeclaredConsumer();
    } else if (!this.state.hasFeedbackLoop) {
      this.state.addFeedbackLoop();
    } else if (this.state.pipelineStagesCount < 3) {
      this.state.addPipelineStage();
    } else if (!this.state.activeHighlight) {
      // Show Feature A impact
      const impact = this.state.triggerChangeFeatureA();
      this.showImpactPanel(impact);
    } else {
      // Run Debt Analysis
      this.showDebtReport();
    }
  }

  updateAll() {
    // 1. Statistics
    this.statAccuracy.textContent = `${this.state.accuracy.toFixed(1)}%`;
    this.statDebt.textContent = `${this.state.debtEvents}`;
    this.statNodes.textContent = `${this.state.graph.getAllNodes().length}`;

    // 2. Technical Debt Meter
    const debtInfo = this.state.getDebtLevel();
    this.debtMeterFill.style.width = `${debtInfo.percent}%`;
    this.debtMeterFill.style.backgroundColor = debtInfo.color;
    this.debtMeterLabel.textContent = debtInfo.label;
    this.debtMeterLabel.style.color = debtInfo.color;

    // 3. Button states
    this.btnAddFeature.disabled = this.state.featureCount >= 4;
    this.btnAddFeature.textContent = this.state.featureCount >= 4 ? '+ All Features Added' : `+ Add Feature (${this.state.featureCount}/4)`;

    this.btnAddCorrection.disabled = this.state.hasCorrectionModel;
    this.btnAddConsumer.disabled = this.state.hasUndeclaredConsumer;
    this.btnAddFeedback.disabled = this.state.hasFeedbackLoop;

    this.btnAddPipeline.disabled = this.state.pipelineStagesCount >= 3;
    this.btnAddPipeline.textContent = this.state.pipelineStagesCount >= 3 ? '+ Pipeline Complete' : `+ Pipeline Stage (${this.state.pipelineStagesCount}/3)`;

    // Enable Change Feature A button only if feature A exists
    this.btnChangeFeatureA.disabled = false; // Always clickable; adds feature A if needed

    // 4. Narrator Card
    const n = this.state.currentNarrative;
    this.narratorBadge.textContent = n.badge;
    this.narratorTitle.textContent = `${n.title} — ${n.concept}`;
    this.narratorQuote.textContent = `“${n.quote}”`;
    this.narratorDetail.textContent = n.detail;

    // 5. Easter Egg
    if (this.state.isMaxComplexity()) {
      this.statusEasterEgg.classList.remove('hidden');
    } else {
      this.statusEasterEgg.classList.add('hidden');
    }
  }

  showImpactPanel(impact) {
    if (!impact || !this.impactPanel) return;

    this.impactPanel.classList.remove('hidden');
    this.impactCount.textContent = impact.affectedNodes.length;

    this.impactList.innerHTML = '';
    
    // Group affected nodes by category for clean presentation
    impact.affectedNodes.forEach(node => {
      const li = document.createElement('li');
      li.className = 'impact-item';
      
      const depth = impact.depthMap.get(node.id) || 1;
      const categoryTag = node.type.toUpperCase().replace('_', ' ');

      li.innerHTML = `
        <span class="impact-node-name">${node.label}</span>
        <span class="impact-hop-badge">+${depth} hop</span>
        <span class="impact-category-tag">${categoryTag}</span>
      `;
      this.impactList.appendChild(li);
    });

    if (impact.cycleDetected) {
      const cycleNotice = document.createElement('li');
      cycleNotice.className = 'impact-item impact-cycle-alert';
      cycleNotice.innerHTML = `
        <span class="impact-node-name">↺ Feedback Loop Closed</span>
        <span class="impact-category-tag" style="color: #A94A3A;">RECURSIVE DRIFT</span>
      `;
      this.impactList.appendChild(cycleNotice);
    }

    this.renderer.render();
  }

  hideImpactPanel() {
    if (this.impactPanel) {
      this.impactPanel.classList.add('hidden');
      this.renderer.render();
    }
  }

  showDebtReport() {
    const report = this.analyzer.analyze();
    this.reportModal.classList.remove('hidden');

    let html = `
      <div class="report-header-section">
        <div class="report-timestamp">TIMESTAMP: ${report.timestamp} | TOTAL AUDITED NODES: ${this.state.graph.getAllNodes().length}</div>
        <div class="report-summary-counts">
          <span class="count-tag high">${report.highCount} HIGH RISK</span>
          <span class="count-tag medium">${report.mediumCount} MEDIUM RISK</span>
        </div>
      </div>
      <div class="report-notice">
        <strong>NOTE:</strong> Illustrative debt analysis — proposed extension inspired by Sculley et al. (NeurIPS 2015).
      </div>
      <div class="report-findings-list">
    `;

    if (report.findings.length === 0) {
      html += `
        <div class="report-finding-empty">
          <div class="finding-severity green">CLEAN</div>
          <div class="finding-title">NO ACCUMULATED DEBT DETECTED</div>
          <p>System maintains minimal boundaries with 1-to-1 data-model-prediction flow.</p>
        </div>
      `;
    } else {
      report.findings.forEach(f => {
        const sevClass = f.severity.toLowerCase();
        html += `
          <div class="report-finding-item severity-${sevClass}">
            <div class="finding-top-row">
              <span class="finding-severity ${sevClass}">${f.severity}</span>
              <span class="finding-title">${f.title}</span>
              <span class="finding-paper-ref">${f.paperRef}</span>
            </div>
            <div class="finding-desc">${f.description}</div>
            <div class="finding-detail">${f.detail}</div>
          </div>
        `;
      });
    }

    html += `</div>`;
    this.reportContent.innerHTML = html;
  }

  hideDebtReport() {
    this.reportModal.classList.add('hidden');
  }

  togglePresentationMode() {
    this.state.isPresentationMode = !this.state.isPresentationMode;
    if (this.state.isPresentationMode) {
      this.appContainer.classList.add('presentation-mode');
      this.btnPresentationToggle.textContent = 'Exit Presentation (Esc)';
      this.btnPresentationToggle.classList.add('active');
    } else {
      this.appContainer.classList.remove('presentation-mode');
      this.btnPresentationToggle.textContent = 'Presentation Mode (P)';
      this.btnPresentationToggle.classList.remove('active');
    }
    // Re-render SVG to ensure crisp rendering
    this.renderer.render();
  }
}
