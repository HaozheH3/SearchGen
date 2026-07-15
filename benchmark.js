const tableBody=document.getElementById('table-body');
const tableHeader=document.getElementById('table-header');
const searchInput=document.getElementById('model-search');
const boardSelect=document.getElementById('board-select');
const stratumSelect=document.getElementById('stratum-select');
const stratumControl=document.getElementById('stratum-control');
const statsSummary=document.getElementById('stats-summary');

const components=[
  ['checklist','Checklist'],['rubric_adaptive','Rubric Adaptive'],
  ['prompt_faithfulness','Prompt Faithfulness'],['image_quality','Image Quality'],
  ['text_rendering','Text Rendering'],['ai_naturalness','AI Naturalness'],
  ['composition_and_aesthetics','Composition & Aesthetics'],
  ['physical_plausibility','Physical Plausibility'],
  ['visual_reference_evaluation','Visual Reference'],
];
let manifest,strata,domains,failureModes,rows=[];
let sortKey='overall_9',sortAsc=false;

const esc=(value)=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');
const score=(value)=>value==null?'<span class="na">N/A</span>':Number(value).toFixed(1);
async function getJson(path){const response=await fetch(path);if(!response.ok)throw new Error(`${path}: ${response.status}`);return response.json()}
function byModel(items){return Object.fromEntries(items.map(item=>[item.model_id,item]))}

function breakdownRows(groups){
  return strata.All.map(base=>{
    const row={...base,All:base.overall_9,groupCoverage:{}};
    Object.entries(groups).forEach(([tag,items])=>{
      const item=byModel(items)[base.model_id];
      row[tag]=item?.overall_9??null;
      row.groupCoverage[tag]=item?`${item.n_scored}/${item.n_total}`:'0/0';
    });
    return row;
  });
}
function sourceRows(){
  if(boardSelect.value==='domains')return breakdownRows(domains);
  if(boardSelect.value==='failure_modes')return breakdownRows(failureModes);
  return [...(strata[stratumSelect.value]||[])];
}
function value(row,key){return key.startsWith('component:')?row.components?.[key.slice(10)]??null:row[key]??null}
function arrow(key){return sortKey===key?(sortAsc?' ↑':' ↓'):''}
function header(key,label,className=''){return `<th class="${className}" data-sort="${esc(key)}">${esc(label)}${arrow(key)}</th>`}

function renderHeaders(){
  let html='<th class="rank">#</th>'+header('display_name','Model','model-col');
  if(boardSelect.value==='components'){
    html+=header('overall_9','Overall-9','primary-score')+header('overall_10','Overall-10 · Paper');
    components.forEach(([key,label])=>html+=header(`component:${key}`,label));
    html+=header('coverage','Coverage');
  }else{
    html+=header('All','All 751','primary-score');
    const groups=boardSelect.value==='domains'?domains:failureModes;
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
  return `<tr><td class="rank">${index+1}</td>${modelCell(row)}<td class="primary-score">${score(row.overall_9)}</td><td>${score(row.overall_10)}</td>${componentCells}<td class="${low}" title="Missing policy: ${esc(row.missing_policy)}">${(row.coverage*100).toFixed(1)}%<span class="coverage-count">${row.n_scored}/${row.n_total}</span></td></tr>`;
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
  else{const groups=boardSelect.value==='domains'?domains:failureModes;tableBody.innerHTML=rows.map((row,index)=>breakdownRow(row,index,groups)).join('')}
  const context=boardSelect.value==='components'?`${stratumSelect.value} · ${stratumSelect.value==='All'?manifest.dataset.n_prompts:manifest.partition[stratumSelect.value]} prompts`:`${Object.keys(boardSelect.value==='domains'?domains:failureModes).length} overlapping groups`;
  statsSummary.innerHTML=`Showing <strong>${rows.length}</strong> models · ${esc(context)}`;
}

async function init(){
  try{
    [manifest,strata,domains,failureModes]=await Promise.all([
      getJson('benchmark-data/manifest.json'),getJson('benchmark-data/leaderboard_by_stratum.json'),
      getJson('benchmark-data/leaderboard_by_domain.json'),getJson('benchmark-data/leaderboard_by_failure_mode.json'),
    ]);
    searchInput.addEventListener('input',update);
    boardSelect.addEventListener('change',()=>{stratumControl.hidden=boardSelect.value!=='components';sortKey=boardSelect.value==='components'?'overall_9':'All';sortAsc=false;update()});
    stratumSelect.addEventListener('change',()=>{sortKey='overall_9';sortAsc=false;update()});
    update();
  }catch(error){console.error(error);tableBody.innerHTML='<tr><td class="message-cell" colspan="100">Unable to load benchmark data.</td></tr>';statsSummary.textContent='Data unavailable'}
}
init();
