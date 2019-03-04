<template>
  <div>
    <section class="p-5 tm-container-outer tm-bg-gray">
        <div class="container">
            <div class="row">
                <div class="col-xs-12 mx-auto tm-about-text-wrap text-center">
                    <h2 class="text-uppercase mb-4">Movie recommendations for you</h2>
                    <p class="mb-4">These are some movies which you may like. In case you have seen any of these movies, please rate it.</p>
                </div>
            </div>
        </div>
    </section>
    <div class="tm-container-outer" id="tm-section-3">
        <ul class="nav nav-pills tm-tabs-links">
            <li v-for="genre in ugenres" class="tm-tab-link-li">
                <a :href="'#'+genre" data-toggle="tab" class="tm-tab-link"
                @click="getRecommendationsforGenre(genre)">
                    {{ genre }}
                </a>
            </li>
        </ul>
      </div>
    <div class="tab-content clearfix">
        <!-- Tab 1 -->
        <div v-for="genre in ugenres" class="tab-pane fade" :id="genre">
          <div v-for="movie in displayMovies" class="tm-recommended-place-wrap">
              <div class="tm-recommended-place">
                  <img :src="movie.imgpath" alt="Image" class="img-fluid tm-recommended-img">
                  <div class="tm-recommended-description-box">
                      <h3 class="tm-recommended-title">{{ movie.title }}</h3>
                      <p class="tm-text-highlight">{{ movie.genres }}</p>
                      <p class="tm-text-gray">{{ movie.overview }}</p>
                  </div>
                  <a href="javascript:void(0)" class="tm-recommended-price-box">
                      <p class="tm-recommended-price">{{ movie.predictedrating }}</p>
                      <p class="tm-recommended-price-link">Predicted Rating</p>
                  </a>
              </div>
        </div>
        <div v-if="displayMovies.length === 0">
          Try Other Genres!
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import StarRating from 'vue-star-rating'
export default {
  'name': 'recommendations',
  components:{
    StarRating
  },
  props: ['recommendations', 'getRecommendations'],

  data(){
    return{
      urecomm: this.recommendations,
      uratings: [],
      selectedGenre: 'All',
      ugenres: ['Comedy', 'Sci-Fi', 'Adventure', 'Drama', 'Thriller', 'Horror', 'Action', 'All'],
      displayMovies: [],
    }
  },
  watch: {
    recommendations: function(val){
      this.urecomm = val
      if( this.selectedGenre === 'All'){
        this.getRecommendationsforGenre(this.selectedGenre)
      }
      else{
        this.getTopRatedMovies()
      }
    }
  },
  methods: {
    setRating: function(rating, id){
      document.getElementById("moviename").value = ''
      if (rating > 0){
        this.uratings.push({"movieId":id, "rating":rating});
        this.getRecommendations(this.uratings)
      }
      for(let child in this.$children){
        this.$children[child].fillLevel = [0, 0, 0, 0, 0]
        this.$children[child].currentRating = 0;
      }
    },
    getRecommendationsforGenre(genre){
      this.selectedGenre = genre
      this.getTopRatedMovies()
    },
    getTopRatedMovies(){
      let vm = this;
      this.displayMovies = Array.from(Object.create(this.urecomm));
      this.displayMovies.sort(this.sortByRating)
      this.displayMovies = this.displayMovies.filter(function(x){
        return x.userrating <= 0 && !x.hasOwnProperty('skip')
      })
      if (vm.selectedGenre !== 'All'){
        this.displayMovies = this.displayMovies.filter(function(x){
          return x.genres.includes(vm.selectedGenre)
        })
      }
      this.displayMovies = this.displayMovies.slice(0, 10)
      for(let idx in this.displayMovies){
        let movie = this.displayMovies[idx]
        let url = 'https://api.themoviedb.org/3/movie/'+movie.tmdbId+'?api_key=bb5e2af84af4c7865fc2bb03170ccc42'
        axios.get(url)
        .then(resp => {
          movie.overview = resp.data.overview
          movie.imgpath = 'https://image.tmdb.org/t/p/w300'+resp.data.backdrop_path
        }).catch(error => {
          console.log(error);
        }).then(()=>{
          vm.displayMovies = Object.assign({}, vm.displayMovies, vm.displayMovies)
          for(let child in vm.$children){
            vm.$children[child].fillLevel = [0, 0, 0, 0, 0]
            vm.$children[child].currentRating = 0;
          }
        })
      }
    },
    sortByRating(x, y){
      return y.predictedrating - x.predictedrating
    },
    skipMovie(movie){
      this.uratings.push({"movieId":movie.movieId, "rating":2.5});
      this.getRecommendations(this.uratings)
    }
  }
}
</script>

<style lang="css">
</style>
