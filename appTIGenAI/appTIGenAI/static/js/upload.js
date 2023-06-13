$(document).ready(function() {
    $('#image-container img').each(function() {
        var imageUrl = $(this).attr('src');
        $('<img>').attr('src', imageUrl);
    });
    
});
