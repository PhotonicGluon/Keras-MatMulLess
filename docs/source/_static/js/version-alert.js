// Adapted from https://github.com/anymail/django-anymail/blob/4c443f5515d1d5269a95cb54cf75057c56a3b150/docs/_static/version-alert.js

"use strict";

// Adds admonition for the latest and nightly versions (i.e., the development versions)
function warnOnDevelopmentVersions() {
    if (
        !(
            window.READTHEDOCS_DATA &&
            (window.READTHEDOCS_DATA.version === "latest" || window.READTHEDOCS_DATA.version === "nightly")
        )
    ) {
        // Not on RTD, or not on one of the development versions
        return;
    }

    // Create the warning
    let warning = document.createElement("div");
    warning.setAttribute("class", "admonition danger");
    warning.innerHTML =
        "<p class='first admonition-title'>Caution</p> " +
        "<p class='last'> " +
        "You're reading the documentation for a <strong>development version</strong>. " +
        "Documentation is available for the <a href='/en/stable/'>current stable release</a>, " +
        "or for older versions through the menu at bottom." +
        "</p>";
    warning.querySelector("a").href = window.location.pathname
        .replace("/latest", "/stable")
        .replace("/nightly", "/stable");

    let parent =
        document.querySelector("div.article-container") ||
        document.querySelector("div.body") ||
        document.querySelector("div.document") ||
        document.body;
    parent.insertBefore(warning, parent.firstChild);
}

document.addEventListener("DOMContentLoaded", warnOnDevelopmentVersions);
