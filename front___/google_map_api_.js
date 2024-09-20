async function initMap() {
  const { Map } = google.maps.importLibrary("maps");
  const map = new google.maps.Map(document.getElementById("map"), {
      center: { lat: 33.4711261, lng: 126.5069733 },
      zoom: 20,
      mapTypeId: 'satellite'
  });

  const greens = [
      { label: "E", name: "화순리", lat: 33.2396495, lng: 126.32834346 ,statusText1: "청정",statusText2: "걱정할 필요 없어 보여요!",imageUrl:"https://image-resource.creatie.ai/137179398023283/137179398023285/99ac48f9d187d740f204a72bdaaff08f.png" },
      { label: "F", name: "방두만", lat:33.4317225, lng: 126.9205576 , statusText1: "오염",statusText2: "녹조 상태가 심각합니다", imageUrl: "https://image-resource.creatie.ai/137179398023283/137179398023285/29a78de4cce5e5532f2e62a8e7c740cb.png" }
  ];

  const bounds = new google.maps.LatLngBounds();
  const infoWindow = new google.maps.InfoWindow();

  const input = document.getElementById('search');
  const autocomplete = new google.maps.places.Autocomplete(input);
  autocomplete.bindTo('bounds', map);

  const marker = new google.maps.Marker({
      map: map
  });

  autocomplete.addListener('place_changed', () => {
      infoWindow.close();
      const place = autocomplete.getPlace();

      if (!place.geometry) {
          console.error("Place has no geometry");
          return;
      }

      if (place.geometry.viewport) {
          map.fitBounds(place.geometry.viewport);
      } else {
          map.setCenter(place.geometry.location);
          map.setZoom(17);
      }

      marker.setPosition(place.geometry.location);
      marker.setVisible(true);
      showInfoWindow(map, marker, place);
      infoWindow.setContent(place.name);
      infoWindow.open(map, marker);
  });

  greens.forEach(({ label, name, lat, lng , statusText1, statusText2, imageUrl}) => {
    const greenMarker = new google.maps.Marker({
        position: { lat, lng },
        label,
        map: map
    });
      bounds.extend(greenMarker.position);

      greenMarker.addListener("click", () => {
        map.panTo(greenMarker.getPosition()); // 위치를 가져오기 위해 getPosition() 사용
        map.setZoom(16); // 지도 확대
        showInfoWindow_f_gm(map, greenMarker,name,label, statusText1,statusText2,imageUrl );
          
      });
  });

  map.fitBounds(bounds);
}

function showInfoWindow(map, marker, place) {
  const infoDiv = document.querySelector('.info');
  if (infoDiv) {

      infoDiv.style.left = '1150px';
      infoDiv.innerHTML = `
          
          <button id="info-button"></button>
          <button id="exit-button"></button>
          <div id="location_txt_big">${place.name}</div>
          <div id="current_status_txt_1">청정</div>
          <div id="current_status_txt_2">걱정할 필요 없어 보여요!</div>
          <img src="https://image-resource.creatie.ai/137179398023283/137179398023285/99ac48f9d187d740f204a72bdaaff08f.png" id="current_status_img"/>
          <button id="capture-button">녹조 현황 분석</button>
          <div id="resultArea"></div>
      `;
      document.getElementById('capture-button').addEventListener('click', function() {
          captureMapImageAndSend();

          const imageUrl = "https://image-resource.creatie.ai/137179398023283/137179398023285/847b2bf0b7204049c9275ef180b491d9.png";

      // 이미지 태그 찾기 또는 새로 생성
      let resultImg = document.getElementById('resultImage');
      if (!resultImg) {
          resultImg = document.createElement('img');
          resultImg.id = 'resultImage';
          resultImg.classList.add('responsive-img'); // CSS 클래스 추가
          document.getElementById('resultArea').appendChild(resultImg);
      }

    // 지정된 이미지 URL로 이미지 태그의 src 속성 업데이트
         setTimeout(() => {
        resultImg.src = imageUrl;
        resultImg.style.display = 'block';  // 지연 후 이미지를 보이게 합니다.
    }, 000)
      });

      document.getElementById('exit-button').addEventListener('click', () => {
          infoDiv.style.left = '1500px';
      });
  }
}
function showInfoWindow_f_gm(map, marker,name,label, statusText1,statusText2,imageUrl) {
  const infoDiv = document.querySelector('.info');
  if (infoDiv) {

      infoDiv.style.left = '1150px';
      infoDiv.innerHTML = `
          
          <button id="info-button"></button>
          <button id="exit-button"></button>
          <div id="location_txt_big">${name}</div>
<div id="current_status_txt_1">${statusText1}</div>
<div id="current_status_txt_2">${statusText2}</div>
<img src="${imageUrl}" id="current_status_img"/>

          <button id="capture-button">녹조 현황 분석</button>
          <div id="resultArea"></div>
      `;
      document.getElementById('capture-button').addEventListener('click', function() {
          captureMapImageAndSend();
          let imageUrl;
  if (label === 'F') {
    imageUrl = "https://image-resource.creatie.ai/137179398023283/137179398023285/d78a5895352937fc493fe48c6d549da2.png";
  } else if (label === 'E') {
    imageUrl = "https://image-resource.creatie.ai/137179398023283/137179398023285/17db79897da1ca1f131b427e52e12d13.png";
  }

          
      // 이미지 태그 찾기 또는 새로 생성
      let resultImg = document.getElementById('resultImage');
      if (!resultImg) {
          resultImg = document.createElement('img');
          resultImg.id = 'resultImage';
          resultImg.classList.add('responsive-img'); // CSS 클래스 추가
          document.getElementById('resultArea').appendChild(resultImg);
      }

    // 지정된 이미지 URL로 이미지 태그의 src 속성 업데이트
    setTimeout(() => {
      resultImg.src = imageUrl;
      resultImg.style.display = 'block';
  }, 2000); // 2초 후 실행
  
      });

      document.getElementById('exit-button').addEventListener('click', () => {
          infoDiv.style.left = '1500px';
      });
  }
}


async function captureMapImageAndSend() {
  const imgSrc = document.getElementById('mapImage').src; // Get the image source URL (map image)

  try {
      const response = await fetch('/upload', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify({ imageUrl: imgSrc }),
      });
      const data = await response.json();
      const resultImg = document.getElementById('resultArea');
      resultImg.src = data.resultUrl; // Set the result image from the server
      resultImg.style.display = 'block'; // Show the result image
  } catch (error) {
      console.error('Error during map capture and send:', error);
  }
}

var currentPageUrl = window.location.href;

// 예를 들어, 이 URL을 콘솔에 출력하거나 어딘가에 사용할 수 있음
console.log(currentPageUrl);
