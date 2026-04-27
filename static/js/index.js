window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");
    });

    // Initialize carousels (default: 1 slide)
    // Exclude LTX-2 carousel (has different settings) and cinema carousels (lazy loaded on tab switch)
    var defaultOptions = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    };
    bulmaCarousel.attach('.carousel:not(#ltx2-carousel):not(#cinema-wan-carousel):not(#cinema-ltx2-carousel):not(#cinema-spf-carousel)', defaultOptions);

    // LTX-2 carousel: show 2 videos at a time
    var ltx2Options = {
        slidesToScroll: 1,
        slidesToShow: 2,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    };
    bulmaCarousel.attach('#ltx2-carousel', ltx2Options);

    // LTX-2 videos: click to unmute/mute only (no pause)
    document.querySelectorAll('#ltx2-carousel video').forEach(function(video) {
        video.style.cursor = 'pointer';

        // Prevent default pause/play on click
        video.addEventListener('click', function(e) {
            var rect = this.getBoundingClientRect();
            var clickY = e.clientY - rect.top;
            var controlsHeight = 45;

            // Only intercept clicks above the controls bar
            if (clickY < rect.height - controlsHeight) {
                e.preventDefault();
                e.stopPropagation();
                // Toggle mute only
                this.muted = !this.muted;
            }
        });

        // Prevent pause on click by immediately resuming
        video.addEventListener('pause', function(e) {
            // Only auto-resume if video should be playing (has autoplay)
            if (this.hasAttribute('autoplay') && !this.ended) {
                var vid = this;
                setTimeout(function() {
                    vid.play();
                }, 10);
            }
        });
    });

    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();
});

// Expose function for re-initialization after dynamic content load
window.reinitializeCarousels = function() {
    var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    };

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
        carousels[i].on('before:show', state => {
            console.log(state);
        });
    }
};

// Cinema Transformations tab switching
$(document).ready(function() {
    // Tab switching for cinema section
    var cinemaTabs = document.querySelectorAll('.tabs ul li[data-target]');
    var cinemaTabContents = document.querySelectorAll('.cinema-tab-content');
    var cinemaCarouselsInitialized = {};

    cinemaTabs.forEach(function(tab) {
        tab.addEventListener('click', function() {
            var target = this.getAttribute('data-target');

            // Update active tab
            cinemaTabs.forEach(function(t) { t.classList.remove('is-active'); });
            this.classList.add('is-active');

            // Show target content, hide others
            cinemaTabContents.forEach(function(content) {
                if (content.id === target) {
                    content.style.display = 'block';

                    // Initialize carousel after display is set to block
                    var carouselEl = content.querySelector('.carousel');
                    if (carouselEl && !cinemaCarouselsInitialized[carouselEl.id]) {
                        // Small delay to ensure DOM is ready
                        setTimeout(function() {
                            bulmaCarousel.attach('#' + carouselEl.id, {
                                slidesToScroll: 1,
                                slidesToShow: 1,
                                loop: true,
                                infinite: true,
                                autoplay: false
                            });
                            cinemaCarouselsInitialized[carouselEl.id] = true;
                        }, 50);
                    }
                } else {
                    content.style.display = 'none';
                }
            });
        });
    });

    // Initialize the first (WAN) carousel on page load
    var firstCinemaCarousel = document.querySelector('#cinema-wan-carousel');
    if (firstCinemaCarousel) {
        bulmaCarousel.attach('#cinema-wan-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: false
        });
        cinemaCarouselsInitialized['cinema-wan-carousel'] = true;
    }
});
