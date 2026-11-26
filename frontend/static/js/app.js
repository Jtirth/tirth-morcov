const palette = ['#5574e5','#8b71e5','#46a7d5','#38b88c','#e4a749','#e27384','#6d85a9','#79b9aa'];
let order = 2;
const byId = (id) => document.getElementById(id);
const safe = (value) => String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
async function getJSON(url, options) { const response = await fetch(url, options); const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Request failed.'); return data; }
function renderPredictions(data) {
  byId('observationCount').textContent = `${data.observations} observation${data.observations === 1 ? '' : 's'}`;
  byId('contextCaption').textContent = `${data.order === 1 ? 'First' : 'Second'}-order model uses ${data.context.length} previous word${data.context.length === 1 ? '' : 's'}: ${data.context.map(safe).join(' ') || 'no context'}.`;
  byId('distributionContext').textContent = `for “${data.context.map(safe).join(' ') || '—'}”`;
  if (!data.predictions.length) {
    byId('predictions').innerHTML = '<div class="empty-prediction">No observed continuation for this context. Try a phrase from the training corpus.</div>';
    byId('equation').textContent = 'No conditional distribution is available for this context.';
    byId('calcDetails').textContent = 'The training corpus has zero transitions for the requested context; no probability is invented.';
    byId('probabilityLegend').innerHTML = '<div class="legend-empty">No observed transitions for this context.</div>';
    if (window.updateProbabilityChart) window.updateProbabilityChart([], []);
    return;
  }
  byId('predictions').innerHTML = data.predictions.map((item, index) => `<div class="prediction-row"><span class="rank">${String(index+1).padStart(2,'0')}</span><span class="prediction-word">${safe(item.word)}</span><span class="prediction-count">${item.count} count${item.count===1?'':'s'}</span><span class="prediction-percent">${item.percent.toFixed(1)}%</span><div class="bar-track"><div class="bar-fill" style="width:${item.percent}%"></div></div></div>`).join('');
  const total = data.observations;
  const terms = data.predictions.map(item => `${item.word}: ${item.count}/${total} = ${item.probability.toFixed(3)} (${item.percent.toFixed(1)}%)`).join(' · ');
  byId('equation').textContent = `P(next word | ${data.context.join(' ')}) = count(context → word) / ${total}`;
  byId('calcDetails').textContent = terms + (data.predictions.length < 5 ? ' · Top observed continuations shown; full distribution is represented in the chart.' : '');
  const fullDistribution = data.distribution || data.predictions;
  byId('probabilityLegend').innerHTML = fullDistribution.slice(0, 8).map((item, index) => `<div class="prob-item"><span class="prob-swatch" style="background:${palette[index%palette.length]}"></span><span>${safe(item.word)} <small>(${item.count}/${total})</small></span><b>${item.percent.toFixed(1)}%</b></div>`).join('') + (fullDistribution.length > 8 ? `<div class="legend-empty">+ ${fullDistribution.length - 8} additional observed continuations included in the chart.</div>` : '');
  if (window.updateProbabilityChart) window.updateProbabilityChart(fullDistribution.map(item=>item.word), fullDistribution.map(item=>item.probability));
}
async function predict() {
  const button = byId('predictButton'); button.disabled = true; button.textContent = 'Calculating…';
  try { renderPredictions(await getJSON(`/api/predict?text=${encodeURIComponent(byId('contextInput').value)}&order=${order}&limit=5`)); }
  catch(error) { byId('predictions').textContent = error.message; }
  finally { button.disabled = false; button.innerHTML = 'Predict <span>→</span>'; }
}
async function refreshDashboard() {
  const stats = await getJSON('/api/dashboard');
  byId('totalWords').textContent = stats.total_words.toLocaleString(); byId('uniqueWords').textContent = stats.unique_words.toLocaleString();
  byId('firstContexts').textContent = stats.distinct_first_contexts.toLocaleString(); byId('secondContexts').textContent = stats.distinct_second_contexts.toLocaleString();
  byId('transitionCount').textContent = `${stats.first_transition_count.toLocaleString()} first-order`;
  const renderRows = (rows) => rows.map(row => `<tr><td>${safe(row.context)}</td><td>${safe(row.next)}</td><td>${row.count}</td><td>${(row.probability*100).toFixed(1)}%</td></tr>`).join('') || '<tr><td colspan="4">No transitions observed yet.</td></tr>';
  byId('firstTransitions').innerHTML = renderRows(stats.first_transitions);
  byId('secondTransitions').innerHTML = renderRows(stats.second_transitions);
  if (window.updateFrequencyChart) window.updateFrequencyChart(stats.top_words);
  await predict();
}
async function loadComparison() {
  try { const data = await getJSON('/api/compare'); byId('comparisonRows').innerHTML = ['1','2'].map(key=>{const m=data[key];return `<tr><td>${m.order === 1 ? 'First-order' : 'Second-order'}</td><td>${(m.coverage*100).toFixed(1)}% <small>(${m.covered}/${m.test_cases})</small></td><td>${(m.top1*100).toFixed(1)}%</td><td>${(m.top3*100).toFixed(1)}%</td></tr>`}).join(''); }
  catch(error) { byId('comparisonRows').innerHTML = `<tr><td colspan="4">${safe(error.message)}</td></tr>`; }
}
document.querySelectorAll('.segment').forEach(button=>button.addEventListener('click',()=>{order=Number(button.dataset.order);document.querySelectorAll('.segment').forEach(item=>item.classList.toggle('active',item===button));predict()}));
byId('predictButton').addEventListener('click',predict);
byId('contextInput').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.nativeEvent?.isComposing&&event.keyCode!==229) predict()});
byId('customText').addEventListener('input',event=>byId('charCount').textContent=`${event.target.value.length.toLocaleString()} / 100,000`);
byId('trainButton').addEventListener('click',async()=>{const message=byId('trainingMessage');message.className='training-message';message.textContent='Training…';try{const data=await getJSON('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:byId('customText').value})});message.textContent=`${data.message} ${data.stats.total_words} words, ${data.stats.unique_words} unique.`;await refreshDashboard()}catch(error){message.className='training-message error';message.textContent=error.message}});
byId('resetButton').addEventListener('click',async()=>{try{const data=await getJSON('/api/reset',{method:'POST'});byId('trainingMessage').className='training-message';byId('trainingMessage').textContent=data.message;await refreshDashboard()}catch(error){byId('trainingMessage').className='training-message error';byId('trainingMessage').textContent=error.message}});
refreshDashboard().catch(error=>{byId('predictions').textContent=error.message});loadComparison();
