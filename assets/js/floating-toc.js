// 2020-12-04 


// Copy text-Table-of-Contents into #floating-toc

// On cloneNode func: https://developer.mozilla.org/en-US/docs/Web/API/Node/cloneNode
const toc = document.querySelector('#text-Table-of-Contents');
let floatingToc = toc.cloneNode(true);
floatingToc.id = "floating-toc";
document.body.appendChild(floatingToc);


// Toggle floating TOC based on button

tocButton  = document.querySelector('#toc-btn');
// const toc = document.querySelector('#text-Table-of-Contents');

tocButton.addEventListener('click', (e) => {
    if (tocButton.textContent === '+') {
        tocButton.textContent = '-'
        floatingToc.style.display = 'block'
    } else {
        tocButton.textContent = '+'
        floatingToc.style.display = 'none';
    }
})
