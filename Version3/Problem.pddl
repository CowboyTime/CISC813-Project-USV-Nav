non-fluents nf_USV_obstacle_nav_v3 {
    domain = USV_obstacle_nav_v3;

    objects{
      obs : {o1, o2};
      USV : {b1};
    };
    non-fluents {
      //define goal coordinates
      XG = 100;
      YG = 100;
    };

}

instance USV_obstacle_nav_inst_v3 {
    domain = USV_obstacle_nav_v3;
    non-fluents = nf_USV_obstacle_nav_v3;
    init-state{

      //inital positioning of ASV
      xPos(b1) = 0.0;
      yPos(b1) = 0.0;

      //obstacle states
      obsXp(o1) = 60.0;
      obsYp(o1) = 40.0;
      obsV(o1) = 1.0;
      obsHead(o1) = 3.1415;

      obsXp(o2) = 40.0;
      obsYp(o2) = 60.0;
      obsV(o2) = 0.8;
      obsHead(o2) = 3.1415;

    };
    //horizon = timesteps
    max-nondef-actions = 2;
    horizon = 120;
    discount = 1.0;
}
