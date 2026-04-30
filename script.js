document.addEventListener('DOMContentLoaded', () => {
  // Scroll animations
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        // Animate progress bars inside
        entry.target.querySelectorAll('.progress-bar .fill').forEach(bar => {
          bar.style.width = bar.dataset.width;
        });
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.animate-in').forEach(el => observer.observe(el));

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.closest('.tab-group');
      group.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      group.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });

  // Navbar scroll effect
  const navbar = document.querySelector('.navbar');
  window.addEventListener('scroll', () => {
    navbar.style.background = window.scrollY > 50
      ? 'rgba(10, 14, 26, 0.95)' : 'rgba(10, 14, 26, 0.85)';
  });

  // Smooth scroll nav links
  document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
      }
    });
  });

  // Counter animation for hero stats
  document.querySelectorAll('.counter').forEach(counter => {
    const target = parseFloat(counter.dataset.target);
    const suffix = counter.dataset.suffix || '';
    const decimals = counter.dataset.decimals ? parseInt(counter.dataset.decimals) : 0;
    let current = 0;
    const increment = target / 60;
    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        current = target;
        clearInterval(timer);
      }
      counter.textContent = current.toFixed(decimals) + suffix;
    }, 25);
  });

  // Active section tracking
  const sections = document.querySelectorAll('.section[id]');
  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
      const top = section.offsetTop - 100;
      if (window.scrollY >= top) current = section.getAttribute('id');
    });
    document.querySelectorAll('.nav-links a').forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === '#' + current) link.classList.add('active');
    });
  });

  // Simulator Data
  const sampleTransactions = [
    {
      id: "TXN-8821",
      time: 128683.9,
      amount: 56.52,
      class: 1,
      v1: 4.28, v2: 0.16, v3: -8.64, v4: 0.62, v10: 0.46, v14: -5.34,
      ann: 0.985, cnn: 0.924, ae: 1.45
    },
    {
      id: "TXN-1045",
      time: 42152.1,
      amount: 12.00,
      class: 0,
      v1: -0.21, v2: 0.05, v3: 1.12, v4: -0.45, v10: -0.12, v14: 0.05,
      ann: 0.002, cnn: 0.001, ae: 0.08
    },
    {
      id: "TXN-7732",
      time: 117538.6,
      amount: 208.17,
      class: 1,
      v1: -13.87, v2: -8.59, v3: -12.57, v4: -0.31, v10: -0.41, v14: -9.85,
      ann: 0.998, cnn: 0.975, ae: 2.12
    },
    {
      id: "TXN-5501",
      time: 85210.4,
      amount: 450.00,
      class: 0,
      v1: 1.15, v2: -0.32, v3: 0.85, v4: 0.21, v10: 0.05, v14: -0.15,
      ann: 0.015, cnn: 0.008, ae: 0.12
    }
  ];

  const transactionList = document.getElementById('transaction-list');
  const predictionEmpty = document.getElementById('prediction-empty');
  const predictionContent = document.getElementById('prediction-content');
  const featureGrid = document.getElementById('feature-grid');
  const btnPredict = document.getElementById('btn-predict');
  const loadingArea = document.getElementById('prediction-loading');
  const resultsArea = document.getElementById('prediction-results-area');
  const finalVerdict = document.getElementById('final-verdict');

  let selectedTxn = null;

  // Initialize Simulator
  if (transactionList) {
    sampleTransactions.forEach(txn => {
      const item = document.createElement('div');
      item.className = 'transaction-item';
      item.innerHTML = `
        <div class="item-header">
          <span class="item-id">${txn.id}</span>
          <span class="item-amount">$${txn.amount.toFixed(2)}</span>
        </div>
        <div class="item-details">
          Time: ${txn.time.toFixed(1)}s • Features: V1, V2, V14...
        </div>
      `;
      item.addEventListener('click', () => {
        document.querySelectorAll('.transaction-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        selectTransaction(txn);
      });
      transactionList.appendChild(item);
    });
  }

  function selectTransaction(txn) {
    selectedTxn = txn;
    predictionEmpty.style.display = 'none';
    predictionContent.style.display = 'block';
    resultsArea.style.display = 'none';
    btnPredict.style.display = 'inline-block';
    finalVerdict.className = 'final-verdict';

    featureGrid.innerHTML = '';
    const features = {
      'Time': txn.time.toFixed(1),
      'Amount': txn.amount.toFixed(2),
      'V1': txn.v1,
      'V2': txn.v2,
      'V3': txn.v3,
      'V4': txn.v4,
      'V10': txn.v10,
      'V14': txn.v14
    };

    for (const [name, val] of Object.entries(features)) {
      featureGrid.innerHTML += `
        <div class="feature-tag">
          <span class="feature-name">${name}</span>
          <span class="feature-val">${val}</span>
        </div>
      `;
    }
  }

  if (btnPredict) {
    btnPredict.addEventListener('click', () => {
      if (!selectedTxn) return;

      btnPredict.style.display = 'none';
      loadingArea.style.display = 'flex';
      resultsArea.style.display = 'none';

      setTimeout(() => {
        loadingArea.style.display = 'none';
        resultsArea.style.display = 'block';

        // Update Model Outputs
        const annScore = (selectedTxn.ann * 100).toFixed(2);
        const cnnScore = (selectedTxn.cnn * 100).toFixed(2);
        const aeError = selectedTxn.ae.toFixed(4);

        document.getElementById('res-ann-score').textContent = `${annScore}%`;
        document.getElementById('res-cnn-score').textContent = `${cnnScore}%`;
        document.getElementById('res-ae-error').textContent = aeError;

        const annLabel = selectedTxn.ann > 0.5 ? 'Fraudulent' : 'Legitimate';
        const cnnLabel = selectedTxn.cnn > 0.5 ? 'Fraudulent' : 'Legitimate';
        const aeLabel = selectedTxn.ae > 0.362 ? 'Anomalous' : 'Normal';

        document.getElementById('res-ann-label').textContent = annLabel;
        document.getElementById('res-ann-label').style.color = selectedTxn.ann > 0.5 ? 'var(--accent-red)' : 'var(--accent-green)';
        document.getElementById('res-cnn-label').textContent = cnnLabel;
        document.getElementById('res-cnn-label').style.color = selectedTxn.cnn > 0.5 ? 'var(--accent-red)' : 'var(--accent-green)';
        document.getElementById('res-ae-label').textContent = aeLabel;
        document.getElementById('res-ae-label').style.color = selectedTxn.ae > 0.362 ? 'var(--accent-red)' : 'var(--accent-green)';

        // Final Verdict
        if (selectedTxn.class === 1) {
          finalVerdict.textContent = 'VERDICT: FRAUD DETECTED';
          finalVerdict.className = 'final-verdict verdict-fraud';
        } else {
          finalVerdict.textContent = 'VERDICT: LEGITIMATE';
          finalVerdict.className = 'final-verdict verdict-legit';
        }
      }, 1500);
    });
  }
});
