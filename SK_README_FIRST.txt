1. _pages: Contain files such as podcasts.html, teaching.html, software.html and publication.html 

2. _data: Contains content
  -- software.yml
  -- podcsts.yml
  -- teaching.yml
  -- talks.yml
  -- navigation.yml
  
3. How to do a blogpost
  -- Create an RMD file in the _Rmd folder
  -- Copy yml header from previous RMD
  -- output: 
  html_document:
    fig_path: "/assets/images/posts/"
    self_contained: true
    keep_md: true
  -- In R set up
  knitr::opts_chunk$set(
  fig.path = "../assets/images/posts/dobin_time_series/",  # Where to save files
  fig.cap = " ",
  fig.align = 'center'
  )
  -- After knitting, move .md file to _posts folder *********** (VERY IMPORTANT)