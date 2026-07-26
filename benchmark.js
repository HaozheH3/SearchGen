const tableBody=document.getElementById('table-body');
const tableHeader=document.getElementById('table-header');
const searchInput=document.getElementById('model-search');
const subsetSelect=document.getElementById('subset-select');
const boardSelect=document.getElementById('board-select');
const stratumSelect=document.getElementById('stratum-select');
const stratumControl=document.getElementById('stratum-control');
const statsSummary=document.getElementById('stats-summary');
const domainGlossary=document.getElementById('domain-glossary');
const failureGlossary=document.getElementById('failure-glossary');
const excludedModels=new Set();

const components=[
  ['checklist','Checklist'],['rubric_adaptive','Rubric Adaptive'],
  ['prompt_faithfulness','Prompt Faithfulness'],['image_quality','Image Quality'],
  ['text_rendering','Text Rendering'],['ai_naturalness','AI Naturalness'],
  ['composition_and_aesthetics','Composition & Aesthetics'],
  ['physical_plausibility','Physical Plausibility'],
  ['visual_reference_evaluation','Visual Reference'],
  ['text_reference_evaluation','Text Reference'],
];
let manifest,sliceBoards,rows=[];
let sortKey='overall_9',sortAsc=false;
const primaryValue=row=>row[manifest?.scoring?.primary_metric||'overall_9'];
const activeBoard=()=>sliceBoards[subsetSelect.value];
const activePromptCount=()=>manifest.evaluation_slices.named[subsetSelect.value].expected_rows;
const activePartition=()=>manifest.slice_partitions[subsetSelect.value];

const esc=(value)=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');
const score=(value)=>value==null?'<span class="na">N/A</span>':Number(value).toFixed(1);
async function getJson(path){const response=await fetch(path);if(!response.ok)throw new Error(`${path}: ${response.status}`);return response.json()}
function byModel(items){return Object.fromEntries(items.map(item=>[item.model_id,item]))}

function breakdownRows(groups){
  return activeBoard().strata.All.map(base=>{
    const row={...base,All:primaryValue(base),groupCoverage:{}};
    Object.entries(groups).forEach(([tag,items])=>{
      const item=byModel(items)[base.model_id];
      row[tag]=item?primaryValue(item):null;
      row.groupCoverage[tag]=item?`${item.n_scored}/${item.n_total}`:'0/0';
    });
    return row;
  });
}
function sourceRows(){
  const board=activeBoard();
  let result;
  if(boardSelect.value==='domains')result=breakdownRows(board.domains);
  else if(boardSelect.value==='failure_modes')result=breakdownRows(board.failure_modes);
  else result=[...(board.strata[stratumSelect.value]||[])];
  return result.filter(row=>!excludedModels.has(row.model_id));
}
function value(row,key){return key.startsWith('component:')?row.components?.[key.slice(10)]??null:row[key]??null}
function arrow(key){return sortKey===key?(sortAsc?' ↑':' ↓'):''}
function header(key,label,className=''){return `<th class="${className}" data-sort="${esc(key)}">${esc(label)}${arrow(key)}</th>`}

function renderHeaders(){
  let html='<th class="rank">#</th>'+header('display_name','Model','model-col');
  if(boardSelect.value==='components'){
    html+=header(manifest.scoring.primary_metric,manifest.scoring.metric_label,'primary-score');
    components.forEach(([key,label])=>html+=header(`component:${key}`,label));
    html+=header('coverage','Coverage');
  }else{
    html+=header('All',`All ${activePromptCount()}`,'primary-score');
    const board=activeBoard();
    const groups=boardSelect.value==='domains'?board.domains:board.failure_modes;
    Object.keys(groups).sort().forEach(tag=>html+=header(tag,tag));
  }
  tableHeader.innerHTML=html;
  tableHeader.querySelectorAll('[data-sort]').forEach(cell=>cell.addEventListener('click',()=>{
    const key=cell.dataset.sort;
    if(sortKey===key)sortAsc=!sortAsc;else{sortKey=key;sortAsc=key==='display_name'}
    update();
  }));
}
function modelCell(row){const cls=row.type==='Open'?'open':'commercial';return `<td class="model-col"><div class="model-cell"><span class="model-name">${esc(row.display_name)}</span><span class="model-type ${cls}">${esc(row.type)}</span></div></td>`}
function componentRow(row,index){
  const componentCells=components.map(([key])=>`<td>${score(row.components[key])}</td>`).join('');
  const low=row.coverage<.95?'coverage-low':'';
  return `<tr><td class="rank">${index+1}</td>${modelCell(row)}<td class="primary-score">${score(primaryValue(row))}</td>${componentCells}<td class="${low}" title="Missing policy: ${esc(row.missing_policy)}">${(row.coverage*100).toFixed(1)}%<span class="coverage-count">${row.n_scored}/${row.n_total}</span></td></tr>`;
}
function breakdownRow(row,index,groups){const cells=Object.keys(groups).sort().map(tag=>`<td title="Coverage ${esc(row.groupCoverage[tag])}">${score(row[tag])}</td>`).join('');return `<tr><td class="rank">${index+1}</td>${modelCell(row)}<td class="primary-score">${score(row.All)}</td>${cells}</tr>`}

function update(){
  const query=searchInput.value.trim().toLowerCase();
  rows=sourceRows().filter(row=>row.display_name.toLowerCase().includes(query));
  rows.sort((a,b)=>{
    const left=value(a,sortKey),right=value(b,sortKey);
    if(left==null&&right==null)return a.display_name.localeCompare(b.display_name);
    if(left==null)return 1;if(right==null)return-1;
    if(typeof left==='string'){const result=left.localeCompare(right);return sortAsc?result:-result}
    return sortAsc?left-right:right-left;
  });
  renderHeaders();
  if(!rows.length)tableBody.innerHTML='<tr><td class="message-cell" colspan="100">No matching models.</td></tr>';
  else if(boardSelect.value==='components')tableBody.innerHTML=rows.map(componentRow).join('');
  else{const board=activeBoard();const groups=boardSelect.value==='domains'?board.domains:board.failure_modes;tableBody.innerHTML=rows.map((row,index)=>breakdownRow(row,index,groups)).join('')}
  const context=boardSelect.value==='components'?`${stratumSelect.value} · ${stratumSelect.value==='All'?activePromptCount():activePartition()[stratumSelect.value]} prompts`:`${Object.keys(boardSelect.value==='domains'?activeBoard().domains:activeBoard().failure_modes).length} overlapping groups`;
  statsSummary.innerHTML=`Showing <strong>${rows.length}</strong> models · ${esc(context)}`;
  domainGlossary.hidden=boardSelect.value!=='domains';
  failureGlossary.hidden=boardSelect.value!=='failure_modes';
}

function updateStratumLabels(){
  const counts=activePartition();
  const labels={
    All:['All',activePromptCount()],
    NoSearch:['NoSearch',counts.NoSearch],
    SearchIntensive:['SearchIntensive',counts.SearchIntensive],
    VisualSearch:['VisualSearch',counts.VisualSearch],
    TextualSearch:['TextualSearch',counts.TextualSearch],
  };
  Array.from(stratumSelect.options).forEach(option=>{
    const [label,count]=labels[option.value];
    option.textContent=`${label} · ${count}`;
  });
}

async function init(){
  try{
    [manifest,sliceBoards]=await Promise.all([
      getJson('benchmark-data/manifest.json'),getJson('benchmark-data/leaderboard_by_slice.json'),
    ]);
    searchInput.addEventListener('input',update);
    sortKey=manifest.scoring.primary_metric;
    subsetSelect.addEventListener('change',()=>{sortKey=boardSelect.value==='components'?manifest.scoring.primary_metric:'All';sortAsc=false;updateStratumLabels();update()});
    boardSelect.addEventListener('change',()=>{stratumControl.hidden=boardSelect.value!=='components';sortKey=boardSelect.value==='components'?manifest.scoring.primary_metric:'All';sortAsc=false;update()});
    stratumSelect.addEventListener('change',()=>{sortKey=manifest.scoring.primary_metric;sortAsc=false;update()});
    updateStratumLabels();
    update();
  }catch(error){console.error(error);tableBody.innerHTML='<tr><td class="message-cell" colspan="100">Unable to load benchmark data.</td></tr>';statsSummary.textContent='Data unavailable'}
}
init();
