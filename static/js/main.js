$(document).ready(function () {
    // стандартная инициализация
    initializeUI();

    // контроллеры управляемых элементов на странице (кнопки, всплывашка с загрузчиком)
    $("#imageUpload").change(handleFileUpload);

    $('#btn-predict').click(handlePrediction);


    function initializeUI() {
        $('.image-section').hide();
        $('.loader').hide();
        $('#result').hide();
    }

    // контроллер загрузки файлов
    function handleFileUpload() {
        const fileInput = this;
        
        if (fileInput.files && fileInput.files[0]) {
            // отображение доступных элементов на текущий момент времени
            $('.image-section').show();
            $('#btn-predict').show(); 
            $('#result').text('').hide();

            readImageFile(fileInput);
        }
    }

    function readImageFile(input) {
        const reader = new FileReader();
        reader.onload = function (e) {
            $('#imagePreview').css('background-image', `url(${e.target.result})`).hide().fadeIn(650);
        };
        reader.readAsDataURL(input.files[0]);
    }

    // функция для обработки картинки
    function handlePrediction() {
        const formData = new FormData($('#upload-file')[0]);

        toggleLoader(true);

        // AJAX вызов, делает POST запрос на эндпоинт, откуда получает ответ
        $.ajax({
            type: 'POST',
            url: '/predict-label',
            data: formData,
            contentType: false,
            cache: false,
            processData: false,
            success: displayPredictionResult,
            error: handleError
        });
    }

    function toggleLoader(showLoader) {
        if (showLoader) {
            $('#btn-predict').hide();
            $('.loader').show();
        } else {
            $('.loader').hide();
            $('#btn-predict').show();
        }
    }

    // функция для отображения результатов обработки картинки
    function displayPredictionResult(data) {
        toggleLoader(false);
        $('#result').fadeIn(600).html(`<strong>Result of the prediction:</strong> ${data}`);
        console.log('Prediction successful!');
    }

    // функция для выброса ошибки в случае если предсказание оказалось невалидным
    function handleError() {
        toggleLoader(false);
        $('#result').text('An error occurred, please try again.').show();
        console.error('Prediction failed.');
    }
});
