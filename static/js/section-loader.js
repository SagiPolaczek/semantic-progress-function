/**
 * Section Loader - Dynamically loads modular HTML sections
 * ReTime Project
 */

// Section configuration
const SECTIONS = [
    { id: 'hero-section', path: './sections/hero.html' },
    { id: 'problem-section', path: './sections/problem.html' },
    { id: 'method-section', path: './sections/method.html' },
    { id: 'gallery-section', path: './sections/gallery.html' },
    { id: 'spf-viz-section', path: './sections/spf-visualization.html' },
    { id: 'ltx2-section', path: './sections/ltx2.html' },
    { id: 'applications-section', path: './sections/applications.html' },
    { id: 'abstract-section', path: './sections/abstract.html' },
    { id: 'bibtex-section', path: './sections/bibtex.html' }
];

/**
 * Fetch and inject section HTML into a placeholder element
 */
async function loadSection(sectionId, htmlPath) {
    try {
        const response = await fetch(htmlPath);
        if (!response.ok) {
            throw new Error(`Failed to load ${htmlPath}: ${response.status}`);
        }
        const html = await response.text();
        const container = document.getElementById(sectionId);
        if (container) {
            container.innerHTML = html;
        } else {
            console.warn(`Container #${sectionId} not found`);
        }
    } catch (error) {
        console.error(`Error loading section ${sectionId}:`, error);
    }
}

/**
 * Load all sections sequentially
 */
async function loadAllSections() {
    for (const section of SECTIONS) {
        await loadSection(section.id, section.path);
    }
}

/**
 * Initialize carousels after sections are loaded
 */
function initializeCarousels() {
    const options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    };

    // Initialize all div with carousel class
    if (typeof bulmaCarousel !== 'undefined') {
        bulmaCarousel.attach('.carousel', options);
    }

    // Initialize sliders
    if (typeof bulmaSlider !== 'undefined') {
        bulmaSlider.attach();
    }
}

/**
 * Initialize problem section carousel
 */
function initializeProblemCarousel() {
    const carousel = document.getElementById('problem-carousel');
    if (!carousel) return;

    const items = carousel.querySelectorAll('.problem-carousel-item');
    const dots = carousel.querySelectorAll('.problem-dot');
    const prevBtn = carousel.querySelector('.carousel-prev');
    const nextBtn = carousel.querySelector('.carousel-next');
    let currentIndex = 0;

    function showItem(index) {
        items.forEach((item, i) => {
            item.classList.toggle('active', i === index);
        });
        dots.forEach((dot, i) => {
            dot.classList.toggle('active', i === index);
        });
        currentIndex = index;
    }

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            const newIndex = (currentIndex - 1 + items.length) % items.length;
            showItem(newIndex);
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            const newIndex = (currentIndex + 1) % items.length;
            showItem(newIndex);
        });
    }

    dots.forEach((dot, index) => {
        dot.addEventListener('click', () => showItem(index));
    });

    // Show first item
    showItem(0);
}

/**
 * Main initialization on DOM ready
 */
document.addEventListener('DOMContentLoaded', async () => {
    // Load all sections
    await loadAllSections();

    // Initialize carousels after sections are loaded
    initializeCarousels();
    initializeProblemCarousel();

    console.log('All sections loaded and initialized');
});
