<template>
  <div>
    <section class="p-5 tm-container-outer tm-bg-gray">
        <div class="container">
            <div class="row">
                <div class="col-xs-12 mx-auto tm-about-text-wrap text-center">
                    <h2 class="text-uppercase mb-4">Rate these movies</h2>
                    <p class="mb-4">Rate some of the movies below. In case you have not seen any movie you can skip it.</p>
                </div>
            </div>
        </div>
    </section>

    <div class="tm-container-outer" id="tm-section-2">
        <section v-for="(movie, idx) in displayMovies"
        :class="{'tm-slideshow-section': idx % 2 === 0, 'clearfix tm-slideshow-section tm-slideshow-section-reverse': idx % 2 !== 0 }">
          <div :class="{'tm-slideshow': idx % 2 === 0, 'tm-right tm-slideshow tm-slideshow-highlight': idx % 2 !== 0 }">
              <img :src="movie.imgpath" alt="Image" style="margin-left: -5%">
          </div>
          <div :class="{'tm-slideshow-description tm-bg-primary': idx % 2 === 0, 'tm-slideshow-description tm-slideshow-description-left tm-bg-highlight': idx % 2 !== 0 }">
              <h2 class="">{{ movie.title }}</h2>
              <h3>{{ movie.genres }}</h3>
              <p>{{ movie.overview }}</p>
              <star-rating :increment="0.5" :rating = "movie.userrating" @rating-selected ="setRating($event, movie.movieId)"></star-rating>
              <br>
              <a href="#ratemovie" class="text-uppercase tm-btn tm-btn-white tm-btn-white-primary" @click="skipMovie(movie)">Skip this movie</a>
          </div>
        </section>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import Vue from 'vue'
import ratemovies from '../rateMovies/Ratemovies'
import recommendations from '../recommendations/Recommendations'
import StarRating from 'vue-star-rating'

export default {
  'name': 'ratemovies',
  props: ['recommendations', 'getRecommendations', 'movieName'],
  components:{
    StarRating
  },
  data(){
    return{
      urecomm: this.recommendations,
      uratings: [],
      displayMovies: [],
      rating: 0,
      uname: this.movieName
    }
  },
  watch: {
    recommendations: function(val){
      this.urecomm = val
      this.getTopRatedMovies()
    },
    movieName: function(val){
      this.uname = val
      this.getTopRatedMovies(this.uname);
    }
  },
  methods: {
    setRating: function(rating, id){
      self = this
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
    getTopRatedMovies(name){
      let vm = this;
      this.displayMovies = Array.from(Object.create(this.urecomm));
      this.displayMovies.sort(this.sortByRating)
      this.displayMovies = this.displayMovies.filter(function(x){
        return x.userrating <= 0 && !x.hasOwnProperty('skip')
          && x.genres != 'Documentary'
      })
      if(typeof name !== 'undefined' && name.length > 0){
        this.displayMovies = this.displayMovies.filter(movie => movie.title.toLowerCase().match(name.toLowerCase()))
      }
      this.displayMovies = this.displayMovies.slice(0, 10)
      for(let idx in this.displayMovies){
        let movie = this.displayMovies[idx]
        let url = 'https://api.themoviedb.org/3/movie/'+movie.tmdbId+'?api_key=bb5e2af84af4c7865fc2bb03170ccc42'
        axios.get(url)
        .then(resp => {
          movie.overview = resp.data.overview
          movie.imgpath = 'https://image.tmdb.org/t/p/w780'+resp.data.backdrop_path
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
