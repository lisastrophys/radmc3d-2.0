program test_dust_temperature_lookup
  use, intrinsic :: ieee_arithmetic, only: ieee_next_after
  use montecarlo_module
  implicit none

  double precision :: energy_at,energy_below,energy_above
  double precision :: temp_at,temp_below,temp_above

  db_ntemp = 3
  allocate(db_temp(db_ntemp))
  allocate(db_enertemp(db_ntemp,1))
  db_temp = (/ 1.d1,2.d1,3.d1 /)
  db_enertemp(:,1) = (/ 1.d-3,1.d-2,1.d-1 /)

  energy_at = db_enertemp(2,1)
  energy_below = ieee_next_after(energy_at,0.d0)
  energy_above = ieee_next_after(energy_at,huge(energy_at))

  ! This is the collision that made a logarithmic hunt choose the lower
  ! interval even though the linear energy lies above the table value.
  if(log(energy_above).ne.log(energy_at)) error stop 5

  temp_below = compute_dusttemp_energy_bd(energy_below,1)
  temp_at = compute_dusttemp_energy_bd(energy_at,1)
  temp_above = compute_dusttemp_energy_bd(energy_above,1)

  if(temp_below.gt.temp_at) error stop 1
  if(temp_at.ne.db_temp(2)) error stop 2
  if(temp_above.lt.temp_at) error stop 3
  if(temp_below.lt.db_temp(1).or.temp_above.gt.db_temp(3)) error stop 4

  deallocate(db_enertemp,db_temp)
  write(*,*) 'dust-temperature lookup boundary regression passed'
end program test_dust_temperature_lookup
