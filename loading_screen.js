const loading_screen_tracker = function () {
  var status = {};

  addEventListener('py:progress', function (evt) {
    var loading_target = document.querySelector('#loading_progress');
    var words = evt.detail.toLowerCase().split(' ');

    if (words[0] === 'loading') {
      elt = document.createElement('span');
      status[words[1]] = elt;
      elt.classList.add('tooltip');

      var loader = document.createElement('span');
      loader.innerHTML = '&#9711;';
      elt.appendChild(loader);

      var ttip = document.createElement('span');
      ttip.innerHTML = words[1];
      ttip.classList.add('tooltiptext');
      elt.appendChild(ttip);

      loading_target.appendChild(elt);
    } else {
      elt = status[words[1]];
      elt.children[0].innerHTML = '&#11044;';
    }
  });
};

loading_screen_tracker();
