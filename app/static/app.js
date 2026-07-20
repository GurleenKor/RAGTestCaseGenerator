const form = document.getElementById('generate-form');
const requirementInput = document.getElementById('requirement');
const fileInput = document.getElementById('requirement-file');
const modelInput = document.getElementById('model-name');
const outputCard = document.getElementById('output-card');
const statusPill = document.getElementById('status-pill');

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const requirement = requirementInput.value.trim();
  const file = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
  if (!file && !requirement) {
    renderMessage('Please upload a requirement document or enter requirement text.', 'error');
    setStatus('error');
    return;
  }

  setStatus('loading');
  renderMessage('Generating test cases...', 'loading');

  try {
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    }
    if (requirement) {
      formData.append('requirement', requirement);
    }
    const modelName = modelInput.value.trim();
    if (modelName) {
      formData.append('model_name', 'qwen2.5-coder:latest');
    }

    const response = await fetch('/api/generate-testcases', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Request failed');
    }

    const text = data.test_cases || 'No test cases returned.';
    renderText(text);
    setStatus('success');
  } catch (error) {
    renderMessage(error.message || 'Unexpected error occurred.', 'error');
    setStatus('error');
  }
});

function renderText(text) {
  outputCard.innerHTML = '';

  const wrapper = document.createElement('div');
  wrapper.className = 'structured-output';

  const table = document.createElement('table');
  table.className = 'test-case-table';

  // Table Header
  const headers = [
    'Test Case ID',
    'Scenario',
    'Preconditions',
    'Test Steps',
    'Expected Result',
    'Test Type',
    'Grounding Evidence'
  ];

  const headerRow = document.createElement('tr');

  headers.forEach(header => {
    const th = document.createElement('th');
    th.textContent = header;
    headerRow.appendChild(th);
  });

  table.appendChild(headerRow);

  const lines = text.split('\n');

  lines.forEach(line => {
    const trimmed = line.trim();

    // Skip empty lines
    if (!trimmed) return;

    // Skip separator row
    if (/^\|?[-:\s|]+\|?$/.test(trimmed)) return;

    // Only process markdown table rows
    if (!trimmed.startsWith('|')) return;

    const columns = trimmed
      .split('|')
      .map(col => col.trim())
      .filter(col => col !== '');

    // Skip markdown header from LLM
    if (columns[0] === 'Test Case ID') return;

    const row = document.createElement('tr');

    columns.forEach(col => {
      const td = document.createElement('td');

      // Preserve HTML line breaks from LLM
      td.innerHTML = col.replace(/<br\s*\/?>/gi, '<br>');

      row.appendChild(td);
    });

    // Ensure exactly 7 columns
    while (row.children.length < 7) {
      const td = document.createElement('td');
      td.textContent = 'Not specified';
      row.appendChild(td);
    }

    // Ignore extra columns if any
    while (row.children.length > 7) {
      row.removeChild(row.lastChild);
    }

    table.appendChild(row);
  });

  wrapper.appendChild(table);
  outputCard.appendChild(wrapper);
}



function renderMessage(message, type) {
  outputCard.innerHTML = '';
  const paragraph = document.createElement('p');
  paragraph.className = 'empty-state';
  paragraph.textContent = message;
  outputCard.appendChild(paragraph);
  outputCard.dataset.state = type;
}

function setStatus(state) {
  statusPill.className = `status-pill ${state}`;
  const labels = {
    idle: 'Idle',
    loading: 'Generating',
    success: 'Success',
    error: 'Error',
  };
  statusPill.textContent = labels[state] || 'Idle';
}
