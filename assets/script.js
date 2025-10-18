const API_BASE_URL = 'http://localhost:5001/api';
let currentPredictions = null;
let systemStatus = { connected: false, trained: false };

function el(id){ return document.getElementById(id); }

async function checkSystemStatus(){
  try{
    const res = await fetch(`${API_BASE_URL}/status`);
    const data = await res.json();
    if(data.success){
      systemStatus.connected = true;
      systemStatus.trained = data.models_trained;
      updateStatusIndicator('🟢 Connected', 'status-connected');
      if(!data.models_trained){ showMessage('⚠️ Models not trained. Training now...','error'); await trainModels(); }
    } else throw new Error('Status check failed');
  }catch(err){
    systemStatus.connected = false; updateStatusIndicator('🔴 Disconnected','status-disconnected'); showMessage('❌ Cannot connect to backend. Ensure server is running.','error');
    console.error(err);
  }
}

function updateStatusIndicator(text, className){ const ind = el('statusIndicator'); ind.textContent = text; ind.className = `status-indicator ${className}`; }

async function trainModels(){ try{ updateLoadingStatus('Training AI models...'); const resp = await fetch(`${API_BASE_URL}/train`,{method:'POST',headers:{'Content-Type':'application/json'}}); const data = await resp.json(); if(data.success){ systemStatus.trained = true; showMessage('✅ Models trained!', 'success'); } else throw new Error(data.error||'Training failed'); }catch(err){ showMessage(`❌ Training failed: ${err.message}`,'error'); console.error(err);} }

function showMessage(message, type='info'){ const m = el('messages'); m.innerHTML = `<div class="${type==='error'?'error-message':'success-message'}">${message}</div>`; setTimeout(()=>m.innerHTML='','5000'); }

function updateLoadingStatus(text){ const s = el('loadingStatus'); if(s) s.textContent = text; }

function formatModelName(name){ const names = {'neural_network':'Neural Network','random_forest':'Random Forest','gradient_boost':'Gradient Boost','logistic_regression':'Logistic Regression','svm':'SVM','ensemble':'🌟 Ensemble'}; return names[name]||name; }
function getRiskClass(riskScore){ if(riskScore<0.3) return 'risk-low'; if(riskScore<0.6) return 'risk-medium'; return 'risk-high'; }
function getPerformanceBadgeClass(accuracy){ if(accuracy>=0.9) return 'badge-excellent'; if(accuracy>=0.8) return 'badge-good'; return 'badge-average'; }

function createResultHTML(modelName, prediction){ const rec = prediction.recommendation; return `
  <div class="risk-score">
    <div class="risk-meter">
      <div class="meter-circle">
        <div class="meter-inner">
          <div class="score-value">${prediction.risk_percentage}%</div>
          <div class="score-label">Risk Score</div>
        </div>
      </div>
    </div>
    <h3 class="${getRiskClass(prediction.risk_score)}">${rec.risk_category}</h3>
  </div>

  <div class="model-info">
    <div><strong>${prediction.model_info.model_type}</strong></div>
    <div class="model-accuracy">Accuracy: ${Math.round(prediction.model_info.accuracy*100)}%</div>
  </div>

  <div class="recommendation">
    <h3>💰 Loan Recommendation</h3>
    <div class="rec-item"><span class="rec-label">Maximum Loan Amount:</span><span class="rec-value">₹${rec.max_loan_amount.toLocaleString()}</span></div>
    <div class="rec-item"><span class="rec-label">Interest Rate:</span><span class="rec-value">${rec.interest_rate}%</span></div>
    <div class="rec-item"><span class="rec-label">Approval Probability:</span><span class="rec-value">${rec.approval_probability}%</span></div>
    <div class="rec-item"><span class="rec-label">Recommended Term:</span><span class="rec-value">${rec.recommended_term_months} months</span></div>
    <div class="rec-item"><span class="rec-label">Risk Category:</span><span class="rec-value ${getRiskClass(prediction.risk_score)}">${rec.risk_category}</span></div>
  </div>`; }

function switchTab(modelName){ document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active')); document.querySelectorAll('.model-result').forEach(r=>r.classList.remove('active')); const btn = document.querySelector(`[data-model="${modelName}"]`); if(btn) btn.classList.add('active'); const res = el(`result-${modelName}`); if(res) res.classList.add('active'); }

function getRiskClassName(riskScore){ if(riskScore<0.3) return 'risk-low'; if(riskScore<0.6) return 'risk-medium'; return 'risk-high'; }

function getRiskClassFromScore(score){ if(score<0.3) return 'risk-low'; if(score<0.6) return 'risk-medium'; return 'risk-high'; }

function getRiskClassLabel(score){ if(score<0.3) return 'Low'; if(score<0.6) return 'Medium'; return 'High'; }

function getRiskClass(r){ if(r<0.3) return 'risk-low'; if(r<0.6) return 'risk-medium'; return 'risk-high'; }

function displayResults(predictions){ const tabs = el('modelTabs'); const results = el('modelResults'); const comparison = el('modelComparison'); tabs.innerHTML=''; results.innerHTML=''; comparison.innerHTML=''; let first=true; Object.entries(predictions).forEach(([name,pred])=>{ const tab = document.createElement('button'); tab.className = `tab ${first?'active':''}`; tab.dataset.model = name; tab.textContent = formatModelName(name); tab.onclick = ()=>switchTab(name); const badge = document.createElement('span'); badge.className = `performance-badge ${getPerformanceBadgeClass(pred.model_info.accuracy)}`; badge.textContent = `${Math.round(pred.model_info.accuracy*100)}%`; tab.appendChild(badge); tabs.appendChild(tab); const div = document.createElement('div'); div.className = `model-result ${first?'active':''}`; div.id = `result-${name}`; div.innerHTML = createResultHTML(name,pred); results.appendChild(div); const card = document.createElement('div'); card.className = 'comparison-card'; card.innerHTML = `<div class="comparison-model-name">${formatModelName(name)}</div><div class="comparison-score ${getRiskClass(pred.risk_score)}">${pred.risk_percentage}%</div><div class="comparison-accuracy">Accuracy: ${Math.round(pred.model_info.accuracy*100)}%</div>`; comparison.appendChild(card); first=false; }); }

async function submitAssessment(financialData){ try{ updateLoadingStatus('Running multi-model analysis...'); const resp = await fetch(`${API_BASE_URL}/assess`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(financialData)}); const data = await resp.json(); if(data.success){ currentPredictions = data.predictions; setTimeout(()=>{ displayResults(data.predictions); el('loading').style.display='none'; el('results').style.display='block'; },800); showMessage(`✅ Assessment completed using ${data.model_count} AI models`,'success'); } else { throw new Error(data.error||'Assessment failed'); } }catch(err){ el('loading').style.display='none'; showMessage(`❌ Assessment failed: ${err.message}`,'error'); console.error(err);} }

async function loadHistory(){ try{ const resp = await fetch(`${API_BASE_URL}/history?limit=5`); const data = await resp.json(); if(data.success){ displayHistory(data.history); el('historySection').style.display='block'; el('results').style.display='none'; } else throw new Error(data.error||'History failed'); }catch(err){ showMessage(`❌ Failed to load history: ${err.message}`,'error'); console.error(err);} }

function displayHistory(history){ const cont = el('historyContent'); if(!history || history.length===0){ cont.innerHTML = '<p>No assessment history available.</p>'; return; } cont.innerHTML = history.map(r=>{ const ense = r.predictions.ensemble; const ts = new Date(r.timestamp).toLocaleString(); return `<div class="history-item"><div class="history-timestamp">${ts}</div><div><strong>Risk Score:</strong> ${ense.risk_percentage}% (${ense.recommendation.risk_category})</div><div><strong>Loan Amount:</strong> ₹${ense.recommendation.max_loan_amount.toLocaleString()}</div><div><strong>Input:</strong> Transactions: ${r.input_data.total_transactions}, Payment Score: ${r.input_data.payment_consistency_score}</div></div>`; }).join(''); }

async function exportData(){ try{ const resp = await fetch(`${API_BASE_URL}/export/csv`); const data = await resp.json(); if(data.success){ const blob = new Blob([data.csv_data],{type:'text/csv'}); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href=url; a.download=data.filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); showMessage('✅ Data exported successfully!','success'); } else throw new Error(data.error||'Export failed'); }catch(err){ showMessage(`❌ Export failed: ${err.message}`,'error'); console.error(err);} }

document.addEventListener('DOMContentLoaded', ()=>{
  // wire up form
  const form = el('creditForm'); form.addEventListener('submit', async (e)=>{
    e.preventDefault(); if(!systemStatus.connected){ showMessage('❌ Not connected to backend server','error'); return; } if(!systemStatus.trained){ showMessage('⚠️ Models not trained yet. Please wait...','error'); return; }
    el('loading').style.display='block'; el('results').style.display='none'; el('historySection').style.display='none'; updateLoadingStatus('Collecting financial data...'); const fd = { total_transactions: parseInt(el('totalTransactions').value), avg_transaction_amount: parseFloat(el('avgTransaction').value), payment_consistency_score: parseInt(el('paymentScore').value), business_age_months: parseInt(el('businessAge').value), digital_footprint_score: parseInt(el('digitalScore').value) }; await submitAssessment(fd);
  });

  // buttons
  el('loadHistoryBtn') && (el('loadHistoryBtn').addEventListener('click', loadHistory));
  el('exportBtn') && (el('exportBtn').addEventListener('click', exportData));

  // prefill
  el('totalTransactions').value = '18000'; el('avgTransaction').value='850'; el('paymentScore').value='94'; el('businessAge').value='54'; el('digitalScore').value='88';

  checkSystemStatus();
});

// Theme handling
function applyTheme(theme){
  if(theme === 'dark'){
    document.documentElement.setAttribute('data-theme','dark');
    const btn = el('themeToggle'); if(btn) btn.textContent = '☀️';
  } else {
    document.documentElement.removeAttribute('data-theme');
    const btn = el('themeToggle'); if(btn) btn.textContent = '🌙';
  }
}

function toggleTheme(){
  const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  const next = current === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  try{ localStorage.setItem('theme', next); }catch(e){}
}

// init theme UI
document.addEventListener('DOMContentLoaded', ()=>{
  const stored = (function(){ try{return localStorage.getItem('theme')}catch(e){return null} })();
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(stored || (prefersDark ? 'dark' : 'light'));
  const toggle = el('themeToggle'); if(toggle) toggle.addEventListener('click', toggleTheme);
});
