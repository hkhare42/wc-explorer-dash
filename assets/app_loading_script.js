document.getElementById('start').onclick = function() {
    var newDiv = document.createElement('div');
    newDiv.className = '_dash-loading-callback';
    newDiv.id = 'loading';
    document.body.appendChild(
      newDiv,
      document.getElementById('content'));
  }
  
  document.getElementById('reset').onclick = function() {
    document.getElementById('loading').remove();
  }