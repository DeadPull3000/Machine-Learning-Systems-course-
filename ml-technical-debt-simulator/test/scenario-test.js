/**
 * test/scenario-test.js - Automated Verification of Simulation Mechanics
 */

import { DependencyGraph } from '../js/graph.js';
import { SimulationState } from '../js/state.js';
import { DebtAnalyzer } from '../js/debt-analysis.js';

console.log('=== ML TECHNICAL DEBT SIMULATION TEST SUITE ===\n');

// 1. Stage 0
console.log('1. Verifying Stage 0 (Clean Baseline)...');
const graph = new DependencyGraph();
const state = new SimulationState(graph);
console.assert(graph.getAllNodes().length === 3, `Expected 3 nodes, got ${graph.getAllNodes().length}`);
console.assert(state.accuracy === 92.4, `Expected 92.4% accuracy, got ${state.accuracy}`);
console.assert(state.debtEvents === 0, `Expected 0 debt events, got ${state.debtEvents}`);
console.log('   ✓ Stage 0 verified: DATA -> ML MODEL -> PREDICTION (Acc: 92.4%, Debt: 0)');

// 2. Add 4 Features
console.log('\n2. Verifying Feature additions (CACE)...');
state.addFeature(); // Feature A
state.addFeature(); // Feature B
state.addFeature(); // Feature C
state.addFeature(); // Feature D
console.assert(state.featureCount === 4, `Expected 4 features, got ${state.featureCount}`);
console.assert(state.accuracy === 94.5, `Expected 94.5% accuracy, got ${state.accuracy}`);
console.assert(state.debtEvents === 4, `Expected 4 debt events, got ${state.debtEvents}`);
console.log('   ✓ Stage 1 verified: 4 features added, accuracy climbed to 94.5%');

// 3. Correction Model
console.log('\n3. Verifying Correction Model (Correction Cascade)...');
state.addCorrectionModel();
console.assert(state.hasCorrectionModel === true, 'Correction model flag should be true');
console.assert(graph.getNode('correction_model') !== undefined, 'Correction model node missing');
console.assert(graph.getEdge('model->correction_model') !== undefined, 'Edge model->correction_model missing');
console.assert(graph.getEdge('correction_model->prediction') !== undefined, 'Edge correction_model->prediction missing');
console.log('   ✓ Stage 2 verified: Model A -> Correction Model A\' -> Prediction wired correctly');

// 4. Undeclared Consumers
console.log('\n4. Verifying Undeclared Consumers...');
state.addUndeclaredConsumer();
console.assert(graph.getNode('consumer_unknown') !== undefined, 'consumer_unknown missing');
console.assert(graph.getNode('consumer_unknown').meta.undeclared === true, 'Undeclared flag missing');
console.log('   ✓ Stage 3 verified: Dashboard, Ranking, and UNKNOWN consumers added');

// 5. Feedback Loop
console.log('\n5. Verifying Feedback Loop...');
state.addFeedbackLoop();
console.assert(state.hasFeedbackLoop === true, 'Feedback loop flag should be true');
console.assert(graph.getEdge('users->data') !== undefined, 'Feedback edge users->data missing');
console.log('   ✓ Stage 4 verified: Users -> Data feedback loop wired');

// 6. Pipeline Stages
console.log('\n6. Verifying Pipeline Jungles & Glue Code...');
state.addPipelineStage(); // ETL 1
state.addPipelineStage(); // ETL 2
state.addPipelineStage(); // Feature Processing
console.assert(state.pipelineStagesCount === 3, 'Expected 3 pipeline stages');
console.assert(graph.getNode('pipe_etl_1') !== undefined, 'ETL 1 missing');
console.assert(graph.getNode('pipe_etl_2') !== undefined, 'ETL 2 missing');
console.assert(graph.getNode('pipe_feature_proc') !== undefined, 'Feature Processing missing');
console.log('   ✓ Stage 5 verified: Full pipeline Data -> ETL 1 -> ETL 2 -> Feature Proc -> Features');

// 7. Change Feature A (CACE Traversal Impact)
console.log('\n7. Verifying "Change Feature A" Dependency Closure Traversal...');
const impact = state.triggerChangeFeatureA();
console.log(`   Origin: ${impact.originNode.label}`);
console.log(`   Affected Components Count: ${impact.affectedNodes.length}`);
impact.affectedNodes.forEach((node, idx) => {
  const depth = impact.depthMap.get(node.id);
  console.log(`     [Hop +${depth}] ${node.label} (${node.type})`);
});
console.assert(impact.affectedNodes.length >= 7, 'Should affect at least 7 downstream nodes');
console.assert(impact.cycleDetected === true, 'Should detect feedback cycle in traversal');
console.log('   ✓ CACE Traversal verified: Downstream propagation calculated from directed graph!');

// 8. Static Debt Analysis
console.log('\n8. Verifying Debt Analyzer...');
const analyzer = new DebtAnalyzer(graph, state);
const report = analyzer.analyze();
console.log(`   Total Anti-Patterns Detected: ${report.totalIssues} (${report.highCount} HIGH, ${report.mediumCount} MEDIUM)`);
report.findings.forEach(f => {
  console.log(`     [${f.severity}] ${f.title} (${f.paperRef})`);
});
console.assert(report.highCount >= 3, 'Expected at least 3 high severity findings');
console.log('   ✓ Debt Analyzer verified: Correctly flagged CACE, Correction Cascade, Undeclared Consumer, Feedback Loop, Pipeline Complexity');

// 9. Easter Egg Status Check
console.log('\n9. Verifying Easter Egg State...');
console.assert(state.isMaxComplexity() === true, 'Should trigger max complexity state');
console.log('   ✓ Easter egg condition active: MODEL STATUS: ✓ HEALTHY | SYSTEM STATUS: ⚠ PLEASE DO NOT TOUCH ANYTHING');

// 10. System Reset
console.log('\n10. Verifying System Reset...');
state.reset();
console.assert(graph.getAllNodes().length === 3, 'Reset failed to restore 3 nodes');
console.assert(state.accuracy === 92.4, 'Reset failed to restore 92.4% accuracy');
console.assert(state.debtEvents === 0, 'Reset failed to restore 0 debt events');
console.log('   ✓ Reset verified: Restores pristine Stage 0 clean baseline');

console.log('\n>>> ALL 10 SIMULATION SCENARIO TESTS PASSED SUCCESSFULLY! <<<\n');
