window.addEventListener("resize", handleResponsiveDisplay);
window.addEventListener("DOMContentLoaded", handleResponsiveDisplay);

function handleResponsiveDisplay() {
  var mobileStickyBar = document.getElementById("mobile_sticky_bar");

  if (mobileStickyBar) {
    if (window.innerWidth <= 768) {
      // Mobile screen width threshold
      mobileStickyBar.style.display = "block";
    } else {
      // Desktop view
      mobileStickyBar.style.display = "none";
    }
  }
}


 (function ($) {
    var ost = 0,
        upTotal = 0,
        downTotal = 0;
    $(window).scroll(function () {
        var cOst = $(this).scrollTop();

        if (cOst > ost) {
            upTotal = 0;
            downTotal += (cOst - ost);
            if (downTotal >= 100) {
                $('nav').addClass('fixed');
            }
        } else {
            downTotal = 0;
            upTotal += (ost - cOst);
            if (upTotal >= 100) {
                $('nav').removeClass('fixed');
            }
        }
        ost = cOst;
    });
})(jQuery);
 

const hamburger = document.querySelector(".hamburger");
const navMenu = document.querySelector(".nav-menu");

hamburger.addEventListener("click", () => {

  /* Toggle active class */
  hamburger.classList.toggle("active");
  navMenu.classList.toggle("active");

  /* Toggle aria-expanded value */
  let menuOpen = navMenu.classList.contains("active");
  console.log(menuOpen)
  let newMenuOpenStatus = menuOpen;
  hamburger.setAttribute("aria-expanded", newMenuOpenStatus);
})

// close mobile menu
document.querySelectorAll(".nav-link").forEach(n => n.addEventListener("click", () => {
  hamburger.classList.remove("active");
  navMenu.classList.remove("active");
//   Need to add Toggle aria-expanded value here as well because it stays as true when you click a menu item
}))